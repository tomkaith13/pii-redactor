import logging
import os
import re
from collections import Counter
from typing import Any
from pathlib import Path

import dspy
from datasets import Dataset, load_dataset
from dspy.evaluate.metrics import f1_score

from redactor import PIIRedactor

logger = logging.getLogger(__name__)

DATASET_DIR = "./data/ai4privacy"
OPTIMIZED_MODEL_PATH = "./optimized_model/pii_redactor.json"


def download_dataset(data_dir: str = DATASET_DIR) -> Dataset:
    """Download ai4privacy/pii-masking-300k if not cached locally.

    Returns the English-only train split.
    """
    logger.info("Loading dataset (cache_dir=%s)...", data_dir)
    ds = load_dataset(
        "ai4privacy/pii-masking-300k",
        split="train",
        cache_dir=data_dir,
    )
    english = ds.filter(lambda row: row["language"] == "English")
    logger.info("Filtered to %d English samples", len(english))
    return english


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


def pii_only_f1(gold_text: str, pred_text: str) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 on PII label tokens only.

    Extracts [LABEL] tokens from both texts and computes multiset-based
    precision, recall, and F1.  Ignores surrounding non-PII text entirely,
    so the score reflects actual redaction quality rather than boilerplate
    text matching.

    Returns (precision, recall, f1).
    """
    gold_labels = extract_pii_labels(gold_text)
    pred_labels = extract_pii_labels(pred_text)

    if not gold_labels and not pred_labels:
        return 1.0, 1.0, 1.0
    if not gold_labels or not pred_labels:
        return 0.0, 0.0, 0.0

    gold_counts = Counter(gold_labels)
    pred_counts = Counter(pred_labels)

    tp = sum((gold_counts & pred_counts).values())
    fp = sum((pred_counts - gold_counts).values())
    fn = sum((gold_counts - pred_counts).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _build_feedback(
    gold_text: str, pred_text: str, precision: float, recall: float, f1: float
) -> str:
    """Build detailed, actionable feedback for GEPA reflection.

    Instead of generic "close" / "incorrect" messages, this tells the
    optimizer exactly which PII labels were missed, which were falsely
    added, and which label types were confused.
    """
    if f1 == 1.0 and gold_text.strip() == pred_text.strip():
        return "Correct. All PII entities detected with correct labels."

    gold_labels = extract_pii_labels(gold_text)
    pred_labels = extract_pii_labels(pred_text)
    gold_counts = Counter(gold_labels)
    pred_counts = Counter(pred_labels)

    missed = gold_counts - pred_counts
    extra = pred_counts - gold_counts

    parts = [f"PII-only F1={f1:.2f} (precision={precision:.2f}, recall={recall:.2f})."]

    if missed:
        missed_items = [
            f"{label} (x{count})" if count > 1 else label
            for label, count in missed.items()
        ]
        parts.append(f"Missed PII labels (false negatives): {', '.join(missed_items)}.")

    if extra:
        extra_items = [
            f"{label} (x{count})" if count > 1 else label
            for label, count in extra.items()
        ]
        parts.append(f"Extra PII labels (false positives): {', '.join(extra_items)}.")

    if f1 == 1.0 and gold_text.strip() != pred_text.strip():
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
    """PII-only F1 metric with detailed feedback for GEPA.

    Computes F1 exclusively on [LABEL] tokens extracted from the redacted
    text, so the score reflects entity detection quality rather than being
    inflated by matching non-PII boilerplate.  Feedback enumerates specific
    missed and falsely-added labels to give GEPA actionable reflection
    signals.

    Returns dspy.Prediction(score=float, feedback=str).
    """
    gold_text = gold.redacted_text.strip()
    pred_text = pred.redacted_text.strip()

    precision, recall, f1 = pii_only_f1(gold_text, pred_text)
    feedback = _build_feedback(gold_text, pred_text, precision, recall, f1)

    return dspy.Prediction(score=f1, feedback=feedback)


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
