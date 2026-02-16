import logging
import os
import re
from collections import Counter
from typing import Any
from pathlib import Path

import dspy
from datasets import Dataset, load_dataset, load_from_disk
from dspy.evaluate.metrics import f1_score

from examples import FEWSHOT_ROW_IDS
from redactor import PIIRedactor

logger = logging.getLogger(__name__)

DATASET_DIR = "./data/ai4privacy"
PROCESSED_DATASET_DIR = "./data/ai4privacy_processed"
OPTIMIZED_MODEL_PATH = "./optimized_model/pii_redactor.json"


def download_dataset(
    data_dir: str = DATASET_DIR,
    processed_dir: str = PROCESSED_DATASET_DIR,
) -> Dataset:
    """Download ai4privacy/pii-masking-300k if not cached locally.

    Returns the English-only train split with few-shot rows excluded.
    On first call, downloads from HF Hub, filters, and saves the processed
    dataset to disk.  Subsequent calls load directly from disk without
    contacting HF Hub.
    """
    if Path(processed_dir).exists():
        logger.info("Loading processed dataset from %s", processed_dir)
        return load_from_disk(processed_dir)

    logger.info("Downloading dataset from HF Hub (cache_dir=%s)...", data_dir)
    ds = load_dataset(
        "ai4privacy/pii-masking-300k",
        split="train",
        cache_dir=data_dir,
    )
    english = ds.filter(lambda row: row["language"] == "English")
    logger.info("Filtered to %d English samples", len(english))
    # Exclude few-shot demo rows to prevent data leakage
    excluded = FEWSHOT_ROW_IDS
    filtered = english.filter(lambda row: row["id"] not in excluded)
    logger.info(
        "Excluded %d few-shot rows, %d remaining",
        len(english) - len(filtered),
        len(filtered),
    )
    filtered.save_to_disk(processed_dir)
    logger.info("Saved processed dataset to %s", processed_dir)
    return filtered


def prepare_examples(
    dataset: Dataset,
    train_size: int | None = None,
    val_size: int | None = None,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Convert HF dataset rows to DSPy Examples.

    Takes train_size + val_size samples, maps source_text -> text (input)
    and target_text -> redacted_text (output).
    Sizes default to env vars OPTIMIZE_TRAIN_SIZE / OPTIMIZE_VAL_SIZE (450/50).
    """
    train_size = train_size or int(os.environ.get("OPTIMIZE_TRAIN_SIZE", "450"))
    val_size = val_size or int(os.environ.get("OPTIMIZE_VAL_SIZE", "50"))
    n = train_size + val_size
    subset = dataset.select(range(min(n, len(dataset))))
    examples = [
        dspy.Example(
            text=row["source_text"],
            redacted_text=row["target_text"],
        ).with_inputs("text")
        for row in subset
    ]
    trainset = examples[:train_size]
    valset = examples[train_size:]
    logger.info("Prepared %d train, %d val examples", len(trainset), len(valset))
    return trainset, valset


PII_LABEL_RE = re.compile(r"\[([A-Z]+\d*)\]")


def extract_pii_labels(text: str) -> list[str]:
    """Extract all PII label tokens (e.g. GIVENNAME1, TEL) from redacted text."""
    return PII_LABEL_RE.findall(text)


def hybrid_pii_score(gold_text: str, pred_text: str) -> tuple[float, float, float, int]:
    """Compute hybrid PII detection + classification score.

    Score = 0.75 * detection_recall + 0.25 * classification_accuracy

    Detection recall (label-agnostic): how many gold PII items were
    redacted with ANY label.  Over-redaction is acceptable — no precision
    penalty in the detection component.

    Classification accuracy: of the detected items, how many have the
    exact correct label?

    Returns (detection_recall, classification_accuracy, hybrid_score,
             num_correct_labels).
    """
    gold_labels = extract_pii_labels(gold_text)
    pred_labels = extract_pii_labels(pred_text)

    if not gold_labels and not pred_labels:
        return 1.0, 1.0, 1.0, 0
    if gold_labels and not pred_labels:
        return 0.0, 0.0, 0.0, 0
    if not gold_labels and pred_labels:
        return 1.0, 1.0, 1.0, 0

    gold_counts = Counter(gold_labels)
    pred_counts = Counter(pred_labels)

    total_gold = sum(gold_counts.values())
    total_pred = sum(pred_counts.values())

    # Detection recall: label-agnostic count of redacted items
    detected_count = min(total_gold, total_pred)
    detection_recall = detected_count / total_gold

    # Classification accuracy: exact label matches among detected items
    num_correct = sum((gold_counts & pred_counts).values())
    classification_acc = num_correct / detected_count if detected_count > 0 else 0.0

    DETECTION_WEIGHT = 0.75
    CLASSIFICATION_WEIGHT = 0.25
    hybrid_score = (
        DETECTION_WEIGHT * detection_recall + CLASSIFICATION_WEIGHT * classification_acc
    )

    return detection_recall, classification_acc, hybrid_score, num_correct


def _build_feedback(
    gold_text: str,
    pred_text: str,
    detection_recall: float,
    classification_acc: float,
    hybrid_score: float,
    num_correct: int,
) -> str:
    """Build severity-weighted feedback for GEPA reflection.

    Distinguishes CRITICAL errors (missed PII — under-redaction) from
    minor errors (wrong label on a detected item).  Over-redaction is
    noted but explicitly marked as acceptable.
    """
    if hybrid_score == 1.0 and gold_text.strip() == pred_text.strip():
        return "Correct. All PII entities detected with correct labels."

    gold_labels = extract_pii_labels(gold_text)
    pred_labels = extract_pii_labels(pred_text)
    gold_counts = Counter(gold_labels)
    pred_counts = Counter(pred_labels)

    total_gold = sum(gold_counts.values())
    total_pred = sum(pred_counts.values())
    detected = min(total_gold, total_pred)

    parts = [
        f"Hybrid score={hybrid_score:.2f} "
        f"(detection_recall={detection_recall:.2f}, "
        f"classification_acc={classification_acc:.2f})."
    ]

    # CRITICAL: missed PII (under-detection)
    if detected < total_gold:
        missed_count = total_gold - detected
        parts.append(
            f"CRITICAL: Missed {missed_count} PII item(s) — "
            f"only {detected}/{total_gold} redacted."
        )

    # Minor: wrong labels on detected items
    if detected > 0 and num_correct < detected:
        mislabeled_gold = gold_counts - (gold_counts & pred_counts)
        mislabeled_pred = pred_counts - (gold_counts & pred_counts)
        gold_items = [
            f"{label} (x{c})" if c > 1 else label
            for label, c in mislabeled_gold.items()
        ]
        pred_items = [
            f"{label} (x{c})" if c > 1 else label
            for label, c in mislabeled_pred.items()
        ]
        parts.append(
            f"Minor: {num_correct}/{detected} detected item(s) have correct labels. "
            f"Missing labels: {', '.join(gold_items)}. "
            f"Unexpected labels: {', '.join(pred_items)}."
        )

    # Note: over-redaction (informational, no penalty)
    if total_pred > total_gold:
        extra_count = total_pred - total_gold
        parts.append(f"Note: Over-redacted by {extra_count} item(s) (acceptable).")

    if hybrid_score == 1.0 and gold_text.strip() != pred_text.strip():
        parts.append("All PII labels match, but surrounding text differs.")

    text_f1 = f1_score(pred_text.strip(), gold_text.strip())
    parts.append(f"Full-text token F1={text_f1:.2f} (for reference).")

    parts.append(f"\nExpected:\n{gold_text}\n\nGot:\n{pred_text}")

    return " ".join(parts)


def pii_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any | None = None,
    pred_name: str | None = None,
    pred_trace: Any | None = None,
) -> dspy.Prediction:
    """Hybrid PII metric: detection recall + classification accuracy.

    Score = 0.75 * detection_recall + 0.25 * classification_accuracy.
    Heavily penalises missed PII (under-redaction) while accepting
    over-redaction.  Feedback categorises errors by severity (CRITICAL
    for missed PII, minor for wrong labels) to guide GEPA reflection.

    Returns dspy.Prediction(score=float, feedback=str).
    """
    gold_text = gold.redacted_text.strip()
    pred_text = pred.redacted_text.strip()

    detection_recall, classification_acc, hybrid_score, num_correct = hybrid_pii_score(
        gold_text, pred_text
    )
    feedback = _build_feedback(
        gold_text,
        pred_text,
        detection_recall,
        classification_acc,
        hybrid_score,
        num_correct,
    )

    return dspy.Prediction(score=hybrid_score, feedback=feedback)


def _sum_lm_cost(lm: dspy.LM) -> float:
    """Sum the cost field from an LM's history entries."""
    return sum(entry.get("cost", 0) or 0 for entry in lm.history)


def optimize(api_key: str, model: str, reflection_model: str | None = None) -> None:
    """Run GEPA optimization pipeline.

    1. Downloads/loads dataset
    2. Prepares examples
    3. Configures DSPy LM
    4. Runs GEPA compilation
    5. Saves optimized program to disk
    6. Logs cost breakdown
    """
    lm = dspy.LM(model, api_key=api_key)
    dspy.configure(lm=lm)

    reflection_model = reflection_model or model
    reflection_lm = (
        dspy.LM(reflection_model, api_key=api_key) if reflection_model != model else lm
    )

    dataset = download_dataset()
    trainset, valset = prepare_examples(dataset)

    student = PIIRedactor()

    logger.info("Starting GEPA optimization (auto=light)...")
    logger.info("Student model: %s", model)
    logger.info("Reflection model: %s", reflection_model)
    optimizer = dspy.GEPA(
        metric=pii_metric,
        auto="medium",
        reflection_lm=reflection_lm,
        num_threads=20,
        track_stats=True,
        add_format_failure_as_feedback=True,
    )
    optimized = optimizer.compile(
        student,
        trainset=trainset,
        valset=valset,
    )

    save_dir = Path(OPTIMIZED_MODEL_PATH).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    optimized.save(OPTIMIZED_MODEL_PATH, save_program=False)
    logger.info("Optimized model saved to %s", OPTIMIZED_MODEL_PATH)

    student_cost = _sum_lm_cost(lm)
    reflection_cost = _sum_lm_cost(reflection_lm) if reflection_lm is not lm else 0.0
    total_cost = student_cost + reflection_cost
    logger.info(
        "Optimization cost — Student: $%.4f, Reflection: $%.4f, Total: $%.4f",
        student_cost,
        reflection_cost,
        total_cost,
    )


def load_optimized_model() -> PIIRedactor | None:
    """Load optimized model from disk if it exists.

    Returns None if no optimized model found.
    """
    if not os.path.exists(OPTIMIZED_MODEL_PATH):
        return None

    logger.debug("Loading optimized model from %s", OPTIMIZED_MODEL_PATH)
    redactor = PIIRedactor()
    redactor.load(OPTIMIZED_MODEL_PATH)
    return redactor
