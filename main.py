import os

import dspy
from dotenv import load_dotenv

from redactor import PIIRedactor


def redact(text: str) -> str:
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("DSPY_MODEL", "gemini/gemini-2.0-flash")
    lm = dspy.LM(model, api_key=api_key)
    dspy.configure(lm=lm)
    redactor = PIIRedactor()
    result = redactor(text=text)
    return result.redacted_text


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
    else:
        input_text = "Call John Smith at 555-123-4567"
    print(redact(input_text))
