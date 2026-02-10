import logging
import os

import dspy
from datasets import Dataset

from optimizer import _sum_lm_cost, download_dataset, load_optimized_model, pii_metric
from redactor import PIIRedactor

logger = logging.getLogger(__name__)


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
        num_threads=20,
        display_progress=True,
        display_table=5,
    )
    result = evaluator(redactor)
    score = result.score if hasattr(result, "score") else float(result)
    logger.info("Evaluation score: %.2f", score)

    cost = _sum_lm_cost(lm)
    logger.info("Evaluation cost: $%.4f", cost)
    return score
