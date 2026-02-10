import logging
import os
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


def pii_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any | None = None,
    pred_name: str | None = None,
    pred_trace: Any | None = None,
) -> dspy.Prediction:
    """Token-level F1 metric with feedback for GEPA.

    Compares pred.redacted_text to gold.redacted_text using token-level F1,
    giving partial credit for partially correct redactions.
    Returns dspy.Prediction(score=float, feedback=str).
    """
    score = f1_score(pred.redacted_text.strip(), gold.redacted_text.strip())

    if score == 1.0:
        feedback = "Correct. The redacted text matches exactly."
    elif score > 0.8:
        feedback = (
            f"Close (F1={score:.2f}). Minor differences.\n"
            f"Expected:\n{gold.redacted_text}\n\nGot:\n{pred.redacted_text}"
        )
    else:
        feedback = (
            f"Incorrect (F1={score:.2f}).\n"
            f"Expected:\n{gold.redacted_text}\n\nGot:\n{pred.redacted_text}"
        )

    return dspy.Prediction(score=score, feedback=feedback)


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
        num_threads=4,
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


def prepare_eval_examples(
    dataset: Dataset,
    eval_size: int | None = None,
    offset: int | None = None,
) -> list[dspy.Example]:
    """Pick a held-out evaluation set from the HF dataset.

    Selects eval_size samples starting at offset (after the optimization
    train+val window).  Defaults to env vars EVALUATE_SIZE / EVALUATE_OFFSET.
    If EVALUATE_OFFSET is not set, it defaults to OPTIMIZE_TRAIN_SIZE +
    OPTIMIZE_VAL_SIZE so the eval set never overlaps with optimization data.
    """
    eval_size = eval_size or int(os.environ.get("EVALUATE_SIZE", "500"))
    if offset is None:
        default_offset = int(os.environ.get("OPTIMIZE_TRAIN_SIZE", "450")) + int(
            os.environ.get("OPTIMIZE_VAL_SIZE", "50")
        )
        offset = int(os.environ.get("EVALUATE_OFFSET", str(default_offset)))
    end = min(offset + eval_size, len(dataset))
    subset = dataset.select(range(offset, end))
    examples = [
        dspy.Example(
            text=row["source_text"],
            redacted_text=row["target_text"],
        ).with_inputs("text")
        for row in subset
    ]
    logger.info("Prepared %d eval examples (offset=%d)", len(examples), offset)
    return examples


def evaluate(api_key: str, model: str) -> float:
    """Evaluate the PII redactor on a held-out test set using dspy.Evaluate.

    Uses examples from the HF dataset that are disjoint from the optimization
    train/val split.  Loads the optimized model if available, otherwise falls
    back to the base PIIRedactor.

    Returns the overall score (0-100).
    """
    lm = dspy.LM(model, api_key=api_key)
    dspy.configure(lm=lm)

    dataset = download_dataset()
    eval_set = prepare_eval_examples(dataset)

    redactor = load_optimized_model()
    if redactor is None:
        logger.info("No optimized model found, evaluating base PIIRedactor")
        redactor = PIIRedactor()
    else:
        logger.info("Evaluating optimized model")

    evaluator = dspy.Evaluate(
        devset=eval_set,
        metric=lambda gold, pred, **kw: pii_metric(gold, pred, **kw).score,
        num_threads=4,
        display_progress=True,
        display_table=5,
    )
    result = evaluator(redactor)
    score = result.score if hasattr(result, "score") else float(result)
    logger.info("Evaluation score: %.2f", score)

    cost = _sum_lm_cost(lm)
    logger.info("Evaluation cost: $%.4f", cost)
    return score


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
