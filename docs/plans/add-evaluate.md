# Add `--evaluate` using `dspy.Evaluate` on a held-out test set

## Context

The project uses GEPA optimization with a train/val split (default 450/50 from the HuggingFace dataset), but has no standalone evaluation step. We want to evaluate the model (base or optimized) on a separate held-out set of 500 examples using `dspy.Evaluate`, providing a proper benchmark independent of the optimization data.

## Key constraint

The current `pii_metric` returns `dspy.Prediction(score=float, feedback=str)` (required by GEPA). `dspy.Evaluate` expects a plain float. We'll use a thin wrapper.

## Changes

### 1. `optimizer.py` — Add `prepare_eval_examples()` and `evaluate()`

- **`prepare_eval_examples(dataset, eval_size=500, offset=500)`**: Picks `eval_size` examples starting at index `offset` (after the default optimization set). Converts them to `dspy.Example` the same way `prepare_examples()` does. Returns a single list (no train/val split). `offset` and `eval_size` default to env vars `EVALUATE_OFFSET` / `EVALUATE_SIZE` (defaults: 500 / 500).

- **`evaluate(api_key, model)`**: Wires everything together:
  1. Configures DSPy LM
  2. Downloads dataset
  3. Calls `prepare_eval_examples()`
  4. Loads optimized model (falls back to base `PIIRedactor`)
  5. Creates `dspy.Evaluate(devset=eval_set, metric=..., num_threads=4, display_progress=True, display_table=5)`
  6. The metric wrapper: `lambda gold, pred, **kw: pii_metric(gold, pred, **kw).score`
  7. Calls `evaluator(redactor)` and logs the result score
  8. Logs API cost

### 2. `main.py` — Add `--evaluate` CLI flag

- Add `--evaluate` argument (same pattern as `--optimize`)
- When set: load env, import `evaluate` from optimizer, call it, exit

### 3. `tests/unit/test_optimizer.py` — Add tests

- `TestPrepareEvalExamples`: test that examples are picked from the correct offset, correct count, proper DSPy Example format
- `TestEvaluate`: mock-based test verifying the evaluate function wires up `dspy.Evaluate` correctly

### 4. `CLAUDE.md` — Update CLI docs and project structure

- Add `--evaluate` flag to usage section
- Mention evaluate in project structure

## Files to modify

- `optimizer.py` — add `prepare_eval_examples()`, `evaluate()`
- `main.py` — add `--evaluate` CLI flag
- `tests/unit/test_optimizer.py` — add tests for new functions
- `CLAUDE.md` — update docs

## Verification

```sh
uv run pytest tests/unit/test_optimizer.py -v   # new tests pass
uv run main.py --evaluate                        # runs evaluation on 500 held-out examples
```
