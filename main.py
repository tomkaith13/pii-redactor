import argparse
import logging
import os

import dspy
from dotenv import load_dotenv

from redactor import PIIRedactor

logger = logging.getLogger(__name__)


def redact(text: str) -> str:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("DSPY_MODEL", "gemini/gemini-2.0-flash")
    logger.info("Using model: %s", model)
    logger.info("Input text: %s", text)
    lm = dspy.LM(model, api_key=api_key)
    dspy.configure(lm=lm)

    from optimizer import load_optimized_model

    redactor = load_optimized_model()
    if redactor is None:
        redactor = PIIRedactor()

    result = redactor(text=text)
    logger.debug("Entities found: %s", result.entities)
    logger.debug("Redacted text: %s", result.redacted_text)
    cost = sum(entry.get("cost", 0) or 0 for entry in lm.history)
    logger.debug("Cost: $%.4f", cost)
    return result.redacted_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redact PII from text")
    parser.add_argument("text", nargs="?", default="Call John Smith at 555-123-4567")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show DSPy prompt/response history"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Download dataset and optimize the PII redactor using GEPA",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the PII redactor on a held-out test set",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomly sample evaluation set instead of sequential selection",
    )
    args = parser.parse_args()

    if args.randomize and not args.evaluate:
        parser.error("--randomize requires --evaluate")

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(name)s %(levelname)s: %(message)s")

    if args.optimize:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        model = os.getenv("DSPY_MODEL", "gemini/gemini-2.0-flash")
        reflection_model = os.getenv("GEPA_REFLECTION_MODEL")

        from optimizer import optimize

        optimize(api_key=api_key, model=model, reflection_model=reflection_model)
        raise SystemExit(0)

    if args.evaluate:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        model = os.getenv("DSPY_MODEL", "gemini/gemini-2.0-flash")

        from evaluator import evaluate

        evaluate(api_key=api_key, model=model, randomize=args.randomize)
        raise SystemExit(0)

    result = redact(args.text)
    logger.info("Redacted result: %s", result)

    if args.verbose:
        logger.info("--- DSPy History ---")
        dspy.inspect_history(n=1)
