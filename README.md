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
cp .env.example .env  # then add your real GOOGLE_API_KEY and optionally change DSPY_MODEL
```

## Usage

```python
from main import redact
print(redact("Call John Smith at 555-123-4567"))
# => "Call [GIVENNAME1] [LASTNAME1] at [TEL]"
```

Or from the CLI:

```sh
uv run main.py "Call John Smith at 555-123-4567"
uv run main.py -v "Call John Smith at 555-123-4567"       # + DSPy prompt/response history
uv run main.py --debug "Call John Smith at 555-123-4567"  # + debug logging
uv run main.py --optimize                                # optimize with GEPA (downloads dataset on first run)
uv run main.py --evaluate                                # evaluate on held-out examples via dspy.Evaluate
uv run main.py --evaluate --randomize                    # evaluate on randomly sampled examples
```

## Tests

```sh
uv run pytest                              # all tests (unit + integration)
uv run pytest tests/unit                   # unit tests only (no API calls)
uv run pytest tests/integration            # integration tests only (requires API key)
uv run pytest -m "not integration"         # also works via marker
```

## Linting

Pre-commit hooks run ruff linting and formatting on every commit:

```sh
uv run pre-commit run --all-files   # manual run
```

## Optimization Results

GEPA optimization on 5500 English samples from ai4privacy/pii-masking-300k (5000 train / 500 val):

| Metric | Score | Cost |
|--------|-------|------|
| Token-level F1 (val) | **93.75%** | $1.12 ($1.04 student + $0.08 reflection) |

- Student model: Gemini 2.0 Flash (`gemini/gemini-2.0-flash`)
- Reflection model: Gemini 3 Flash (`gemini/gemini-3-flash-preview`)
- Optimizer: DSPy GEPA (`auto="medium"`, 4 threads)
- Best program found at iteration 7

## Evaluation Results

Held-out evaluation on 3000 examples (disjoint from optimization train/val split):

| Metric | Score | Cost |
|--------|-------|------|
| Token-level F1 | **93.35%** | $0.86 |

- Model: Gemini 2.0 Flash (`gemini/gemini-2.0-flash`)
- Eval set: 3000 examples at offset 6000 from the HF dataset
- 20 threads, optimized model

## Project structure

- `main.py` — `redact()` public API and CLI entry point (with `-v`/`--debug`/`--optimize`/`--evaluate` flags)
- `redactor.py` — `PIIEntity` data model, `IdentifyPII` DSPy signature, `PIIRedactor` module
- `optimizer.py` — GEPA optimization pipeline (dataset download, metric, optimize, load)
- `evaluator.py` — held-out evaluation via `dspy.Evaluate` (dataset prep, evaluate)
- `examples.py` — 25 few-shot `dspy.Example` instances
- `tests/unit/` — structural tests (examples validation, label coverage, data model, CLI/logging, optimizer)
- `tests/integration/` — live redaction tests (require API key)
- `.env` — `GOOGLE_API_KEY`, `DSPY_MODEL`, `GEPA_REFLECTION_MODEL`, `EVALUATE_SEED` (gitignored)
- `.pre-commit-config.yaml` — ruff lint + format hooks
- `data/` — cached HuggingFace dataset (gitignored, created by `--optimize`)
- `optimized_model/` — saved optimized model state (gitignored, created by `--optimize`)
