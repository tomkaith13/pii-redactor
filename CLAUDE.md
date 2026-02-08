# PII Redactor

DSPy-based PII redactor that replaces personally identifiable information in text with type labels (e.g. `[GIVENNAME1]`, `[TEL]`). Uses Gemini via LiteLLM.

## Stack

- Python 3.13, managed by **uv**
- DSPy (ChainOfThought module)
- Gemini 2.0 Flash via Google API key
- Labels follow the [ai4privacy/pii-masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k) dataset schema

## Setup

```sh
uv sync
cp .env.example .env  # then add your real GOOGLE_API_KEY
```

## Usage

```python
from main import redact
print(redact("Call John Smith at 555-123-4567"))
# => "Call [GIVENNAME1] [LASTNAME1] at [TEL]"
```

Or from the CLI:

```sh
uv run python main.py "Call John Smith at 555-123-4567"
```

## Tests

```sh
uv run pytest                              # all tests (unit + integration)
uv run pytest tests/unit                   # unit tests only (no API calls)
uv run pytest tests/integration            # integration tests only (requires API key)
uv run pytest -m "not integration"         # also works via marker
```

## Project structure

- `main.py` — signature, module, 25 few-shot examples, `redact()` API
- `tests/unit/` — structural tests (examples validation, label coverage, data model)
- `tests/integration/` — live redaction tests (require API key)
- `.env` — `GOOGLE_API_KEY` (gitignored)
