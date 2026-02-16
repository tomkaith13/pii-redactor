import logging
import os
import random
from datetime import datetime

import dspy
from datasets import Dataset

from optimizer import _sum_lm_cost, download_dataset, load_optimized_model, pii_metric
from redactor import PIIRedactor

logger = logging.getLogger(__name__)


def prepare_eval_examples(
    dataset: Dataset,
    eval_size: int | None = None,
    offset: int | None = None,
    randomize: bool = False,
) -> list[dspy.Example]:
    """Pick a held-out evaluation set from the HF dataset.

    Selects eval_size samples starting at offset (after the optimization
    train+val window).  Defaults to env vars EVALUATE_SIZE / EVALUATE_OFFSET.
    If EVALUATE_OFFSET is not set, it defaults to OPTIMIZE_TRAIN_SIZE +
    OPTIMIZE_VAL_SIZE so the eval set never overlaps with optimization data.

    When randomize=True, randomly samples eval_size indices from the pool of
    indices after the optimization window instead of picking sequentially.
    Uses EVALUATE_SEED env var for reproducibility if set.
    """
    eval_size = eval_size or int(os.environ.get("EVALUATE_SIZE", "500"))
    exclude_count = int(os.environ.get("OPTIMIZE_TRAIN_SIZE", "450")) + int(
        os.environ.get("OPTIMIZE_VAL_SIZE", "50")
    )

    if randomize:
        pool = range(exclude_count, len(dataset))
        sample_size = min(eval_size, len(pool))
        seed = os.environ.get("EVALUATE_SEED")
        rng = random.Random(int(seed)) if seed is not None else random.Random()
        indices = sorted(rng.sample(pool, sample_size))
        subset = dataset.select(indices)
        logger.info(
            "Prepared %d eval examples (randomized from pool of %d)",
            len(indices),
            len(pool),
        )
    else:
        if offset is None:
            offset = int(os.environ.get("EVALUATE_OFFSET", str(exclude_count)))
        end = min(offset + eval_size, len(dataset))
        subset = dataset.select(range(offset, end))
        logger.info("Prepared %d eval examples (offset=%d)", len(subset), offset)

    examples = [
        dspy.Example(
            text=row["source_text"],
            redacted_text=row["target_text"],
        ).with_inputs("text")
        for row in subset
    ]
    return examples


def evaluate(api_key: str, model: str, randomize: bool = False) -> float:
    """Evaluate the PII redactor on a held-out test set using dspy.Evaluate.

    Uses examples from the HF dataset that are disjoint from the optimization
    train/val split.  Loads the optimized model if available, otherwise falls
    back to the base PIIRedactor.

    Returns the overall score (0-100).
    """
    lm = dspy.LM(model, api_key=api_key)
    dspy.configure(lm=lm)

    dataset = download_dataset()
    eval_set = prepare_eval_examples(dataset, randomize=randomize)

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
        display_table=0,
    )
    result = evaluator(redactor)
    score = result.score if hasattr(result, "score") else float(result)
    cost = _sum_lm_cost(lm)

    logger.info("Evaluation score: %.2f", score)
    logger.info("Evaluation cost: $%.4f", cost)

    if os.environ.get("GENERATE_LOGS", "").lower() in ("1", "true", "yes"):
        _write_eval_log(result, score, cost, lm)

    return score


def _extract_prompt(lm: dspy.LM) -> str:
    """Extract the prompt template from the first LM history entry."""
    if not lm.history:
        return "(no prompt history available)"

    messages = lm.history[0].get("messages", [])
    if not messages:
        return "(no messages in history)"

    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _write_eval_log(result, score: float, cost: float, lm: dspy.LM) -> None:
    """Write per-example evaluation results to a timestamped log file."""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"logs/evaluation_{timestamp}.log"

    sep = "=" * 80
    lines: list[str] = []

    # Write the prompt used at the top of the log
    lines.append(sep)
    lines.append("PROMPT USED (from first example)")
    lines.append(sep)
    lines.append(_extract_prompt(lm))
    lines.append("")

    for i, (example, prediction, ex_score) in enumerate(result.results, 1):
        lines.append(sep)
        lines.append(f"Example {i}/{len(result.results)}  |  Score: {ex_score:.3f}")
        lines.append("-" * 80)
        lines.append(f"TEXT: {example.text}")
        lines.append("-" * 80)
        lines.append(f"GOLD: {example.redacted_text}")
        lines.append("-" * 80)
        lines.append(f"PRED: {prediction.redacted_text}")
        lines.append("-" * 80)
        lines.append("")

    lines.append(sep)
    lines.append(f"OVERALL SCORE: {score:.2f}%  ({len(result.results)} examples)")
    lines.append(f"COST: ${cost:.4f}")
    lines.append(sep)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Evaluation log written to %s", path)
