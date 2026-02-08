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
    redactor = PIIRedactor()
    result = redactor(text=text)
    logger.debug("Entities found: %s", result.entities)
    logger.debug("Redacted text: %s", result.redacted_text)
    return result.redacted_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redact PII from text")
    parser.add_argument("text", nargs="?", default="Call John Smith at 555-123-4567")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show DSPy prompt/response history"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(name)s %(levelname)s: %(message)s")

    result = redact(args.text)
    print(f"Redacted result: {result}")

    if args.verbose:
        print("\n--- DSPy History ---")
        dspy.inspect_history(n=1)
