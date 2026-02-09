# GEPA Optimization for PII Redactor

## Goal

Add a `--optimize` CLI flag that downloads the ai4privacy/pii-masking-300k dataset, runs DSPy GEPA optimization on the PIIRedactor module, saves the optimized model to disk, and auto-loads it on future `redact()` calls.

## Decisions

- **Language filter**: English only
- **Reflection LM**: Same Gemini model (no extra API keys)
- **Metric**: Exact match on redacted text (0/1 score with textual feedback)
- **Data split**: 450 train / 50 val from 500 English samples
- **GEPA intensity**: `auto="light"`
- **Save format**: JSON state-only (`save_program=False`)

## CLI Interface

```sh
# Run optimization (downloads dataset on first run, caches it)
uv run python main.py --optimize

# Normal usage — auto-loads optimized model if available
uv run python main.py "Call John Smith at 555-123-4567"
```

## Architecture

### New file: `optimizer.py`

Contains all optimization logic:

- `download_dataset(data_dir)` — downloads HF dataset with `cache_dir`, filters to English
- `prepare_examples(dataset, n=500)` — converts to DSPy Examples, splits 450/50
- `pii_metric(gold, pred, ...)` — exact match metric returning `dspy.Prediction(score, feedback)`
- `optimize(api_key, model)` — full pipeline: download → prepare → GEPA compile → save
- `load_optimized_model()` — loads saved model from disk, returns `None` if not found

### Changes to `main.py`

- Add `--optimize` CLI flag
- In `redact()`, try `load_optimized_model()` before falling back to default `PIIRedactor()`

### Dataset → DSPy Example mapping

```python
dspy.Example(
    text=row["source_text"],           # input
    redacted_text=row["target_text"],  # expected output
).with_inputs("text")
```

Only `text` and `redacted_text` are mapped. `entities` field is omitted since the exact-match metric doesn't need it.

### Metric

```python
def pii_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = 1 if pred.redacted_text.strip() == gold.redacted_text.strip() else 0
    feedback = "Correct." if score else f"Expected:\n{gold.redacted_text}\n\nGot:\n{pred.redacted_text}"
    return dspy.Prediction(score=score, feedback=feedback)
```

### GEPA Configuration

```python
dspy.GEPA(
    metric=pii_metric,
    auto="light",
    reflection_lm=lm,           # same Gemini model
    num_threads=4,
    track_stats=True,
    add_format_failure_as_feedback=True,
)
```

### Persistence

- Save: `optimized_program.save("./optimized_model/pii_redactor.json", save_program=False)`
- Load: `PIIRedactor().load("./optimized_model/pii_redactor.json")`

## File changes

| File | Change |
|------|--------|
| `optimizer.py` | New — all optimization logic |
| `main.py` | Add `--optimize` flag, load optimized model in `redact()` |
| `pyproject.toml` | Add `datasets` dependency |
| `.gitignore` | Add `data/`, `optimized_model/` |
| `tests/unit/test_optimizer.py` | New — unit tests for optimizer functions |

## New dependency

- `datasets` (HuggingFace datasets library) — for downloading and loading the dataset
