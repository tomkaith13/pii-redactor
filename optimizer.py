import logging
import os
from pathlib import Path

import dspy
from datasets import load_dataset

from redactor import PIIRedactor

logger = logging.getLogger(__name__)

DATASET_DIR = "./data/ai4privacy"
OPTIMIZED_MODEL_PATH = "./optimized_model/pii_redactor.json"


def download_dataset(data_dir: str = DATASET_DIR):
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


def prepare_examples(dataset, n: int = 500) -> tuple[list, list]:
    """Convert HF dataset rows to DSPy Examples.

    Takes the first n samples, maps source_text -> text (input)
    and target_text -> redacted_text (output).
    Returns (trainset of 450, valset of 50).
    """
    subset = dataset.select(range(min(n, len(dataset))))
    examples = [
        dspy.Example(
            text=row["source_text"],
            redacted_text=row["target_text"],
        ).with_inputs("text")
        for row in subset
    ]
    split = int(len(examples) * 0.9)
    trainset = examples[:split]
    valset = examples[split:]
    logger.info("Prepared %d train, %d val examples", len(trainset), len(valset))
    return trainset, valset


def pii_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Exact match metric with feedback for GEPA.

    Compares pred.redacted_text to gold.redacted_text.
    Returns dspy.Prediction(score=0|1, feedback=str).
    """
    score = 1 if pred.redacted_text.strip() == gold.redacted_text.strip() else 0

    if score == 1:
        feedback = "Correct. The redacted text matches exactly."
    else:
        feedback = (
            f"Incorrect. Expected:\n{gold.redacted_text}\n\nGot:\n{pred.redacted_text}"
        )

    return dspy.Prediction(score=score, feedback=feedback)


def optimize(api_key: str, model: str):
    """Run GEPA optimization pipeline.

    1. Downloads/loads dataset
    2. Prepares examples
    3. Configures DSPy LM
    4. Runs GEPA compilation
    5. Saves optimized program to disk
    """
    lm = dspy.LM(model, api_key=api_key)
    dspy.configure(lm=lm)

    dataset = download_dataset()
    trainset, valset = prepare_examples(dataset)

    student = PIIRedactor()

    logger.info("Starting GEPA optimization (auto=light)...")
    optimizer = dspy.GEPA(
        metric=pii_metric,
        auto="light",
        reflection_lm=lm,
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


def load_optimized_model() -> PIIRedactor | None:
    """Load optimized model from disk if it exists.

    Returns None if no optimized model found.
    """
    if not os.path.exists(OPTIMIZED_MODEL_PATH):
        return None

    logger.info("Loading optimized model from %s", OPTIMIZED_MODEL_PATH)
    redactor = PIIRedactor()
    redactor.load(OPTIMIZED_MODEL_PATH)
    return redactor
