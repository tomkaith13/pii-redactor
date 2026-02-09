# GEPA Reflection Model + Cost Tracking

## Summary

Add `GEPA_REFLECTION_MODEL` env var to configure a separate model for GEPA's reflection LM during optimization. Log cost breakdown after optimization runs.

## Changes

### 1. `.env.example`
Add `GEPA_REFLECTION_MODEL=gemini/gemini-2.5-flash` (commented example).

### 2. `main.py`
In the `--optimize` block, read `GEPA_REFLECTION_MODEL` from env, fall back to `DSPY_MODEL`. Pass to `optimize()`.

### 3. `optimizer.py`
- `optimize()` gains `reflection_model: str` parameter
- Creates a separate `dspy.LM(reflection_model, api_key=api_key)` for GEPA's `reflection_lm`
- After optimization completes, sums `cost` from both LMs' `.history` and logs at INFO level:
  ```
  Optimization cost — Student: $X.XX, Reflection: $X.XX, Total: $X.XX
  ```

### 4. `CLAUDE.md`
Document the new env var and cost logging behavior.

## Behavior

| GEPA_REFLECTION_MODEL | Result |
|---|---|
| Not set | Falls back to `DSPY_MODEL` (current behavior) |
| `gemini/gemini-2.5-flash` | Uses 2.5 Flash for reflection, student stays on `DSPY_MODEL` |

Same `GOOGLE_API_KEY` is used for both models.
