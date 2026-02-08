# PII Redactor — Design

## Overview

A DSPy program that takes a sentence and redacts PII using type labels. Uses Gemini models via Google API key.

## Architecture

All code lives in `main.py`. Three components:

1. **`PIIEntity`** — DSPy-compatible structured output for identified PII (value + label pairs).
2. **`PIIRedactor`** — DSPy `Module` using `ChainOfThought`. The LLM first identifies all PII entities as a list of `(value, label)` pairs, then produces the redacted sentence with `[LABEL]` placeholders.
3. **`redact(text: str) -> str`** — Public function that loads `GOOGLE_API_KEY` from `.env`, configures `dspy.LM` with `gemini/gemini-2.0-flash`, runs the module, and returns the redacted string.

## Dependencies

- `dspy` (includes `litellm` for Gemini connectivity)
- `python-dotenv`

## API Key

Loaded from a `.env` file in the project root:

```
GOOGLE_API_KEY=your-key-here
```

## Label Set

Aligned with the [ai4privacy/pii-masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) dataset for future optimization.

| Category | Labels |
|----------|--------|
| Names | `GIVENNAME1`, `GIVENNAME2`, `LASTNAME1`, `LASTNAME2`, `TITLE` |
| Contact | `TEL`, `EMAIL`, `USERNAME` |
| IDs | `SOCIALNUMBER`, `IDCARD`, `DRIVERLICENSE`, `PASSPORT` |
| Location | `STREET`, `BUILDING`, `CITY`, `STATE`, `POSTCODE`, `COUNTRY`, `SECADDRESS` |
| Personal | `SEX`, `BOD`, `PASS` |
| Digital | `IP` |
| Time | `DATE`, `TIME` |

## Output Format

Type-labeled placeholders:

```
Input:  "Call John Smith at 555-123-4567 or john@example.com"
Output: "Call [GIVENNAME1] [LASTNAME1] at [TEL] or [EMAIL]"
```

## Interface

Python function only (no CLI):

```python
from main import redact

result = redact("Call John Smith at 555-123-4567 or john@example.com")
```

## Examples

20+ hand-crafted DSPy examples will be included in the module covering all label types. These serve as static few-shot demos and can later be replaced by optimizer-selected examples.

## Future Optimization Path

1. Load `ai4privacy/pii-masking-300k` from Hugging Face
2. Map `source_text` -> input, `target_text` -> expected output as DSPy `Example` objects
3. Use a DSPy optimizer (`BootstrapFewShot` or `MIPROv2`) to find optimal few-shot examples/instructions
4. Save optimized program with `program.save()`

Out of scope for initial implementation.
