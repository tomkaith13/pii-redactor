# Add `--randomize` suboption for `--evaluate`

## Context

The current `--evaluate` picks evaluation samples **sequentially** starting at an offset after the optimization train+val window (e.g. indices 6000–15999). This means evaluation always runs on the same contiguous slice of the dataset. We want a `--randomize` flag that **randomly samples** from the full dataset (excluding optimization data) to get a more representative evaluation.

## Decisions

- `--randomize` requires `--evaluate` — error if used alone
- Reproducibility via `EVALUATE_SEED` env var (consistent with existing env-var config pattern)
- Sampling pool: indices `train+val..len(dataset)-1` (excludes optimization data)
- Uses a local `random.Random` instance (no global state mutation)

## Changes

### 1. `evaluator.py` — Add randomize support to `prepare_eval_examples()` and `evaluate()`

**`prepare_eval_examples(dataset, eval_size, offset, randomize)`**:
- New param `randomize: bool = False`
- When `randomize=False`: existing sequential behavior unchanged
- When `randomize=True`:
  1. Compute `exclude_count` = `OPTIMIZE_TRAIN_SIZE + OPTIMIZE_VAL_SIZE` (same env var logic as current offset default)
  2. Build pool of valid indices: `range(exclude_count, len(dataset))`
  3. Read `EVALUATE_SEED` env var — if set, create `random.Random(seed)` instance; otherwise unseeded
  4. `random.sample(pool, min(eval_size, len(pool)))` to pick indices
  5. `dataset.select(sorted(sampled_indices))` to fetch rows
  6. Convert to `dspy.Example` list as before

**`evaluate(api_key, model, randomize)`**:
- New param `randomize: bool = False`
- Pass through to `prepare_eval_examples(dataset, randomize=randomize)`

### 2. `main.py` — Add `--randomize` flag and validation

- Add `--randomize` argument (`action="store_true"`, help text)
- After parsing: if `args.randomize and not args.evaluate`, raise `parser.error("--randomize requires --evaluate")`
- Pass `randomize=args.randomize` to `evaluate()`

### 3. `tests/unit/test_evaluator.py` — Add randomize tests

- Test randomize selects correct number of samples
- Test randomize excludes optimization indices (indices 0..exclude-1 never in selected range)
- Test `EVALUATE_SEED` env var produces reproducible results (same seed → same indices)
- Test randomize caps at available pool size when eval_size > pool

### 4. `README.md` / `CLAUDE.md` — Update CLI docs

- Add `--evaluate --randomize` to usage examples
- Mention `EVALUATE_SEED` env var

## Files to modify

- `evaluator.py` — add `randomize` param to `prepare_eval_examples()` and `evaluate()`
- `main.py` — add `--randomize` flag, validation, pass-through
- `tests/unit/test_evaluator.py` — add randomize tests
- `README.md` — update CLI usage
- `CLAUDE.md` — update CLI usage and env var docs

## Verification

```sh
uv run pytest tests/unit/test_evaluator.py -v   # new + existing tests pass
uv run main.py --evaluate --randomize            # random eval run
uv run main.py --randomize                       # should error
```
