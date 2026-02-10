from unittest.mock import MagicMock

import dspy

from optimizer import (
    load_optimized_model,
    pii_metric,
    prepare_eval_examples,
    prepare_examples,
)


class TestPiiMetric:
    def test_exact_match_scores_one(self):
        gold = dspy.Example(redacted_text="Call [GIVENNAME1] at [TEL]")
        pred = dspy.Prediction(redacted_text="Call [GIVENNAME1] at [TEL]")
        result = pii_metric(gold, pred)
        assert result.score == 1
        assert "Correct" in result.feedback

    def test_mismatch_scores_partial(self):
        gold = dspy.Example(redacted_text="Call [GIVENNAME1] at [TEL]")
        pred = dspy.Prediction(redacted_text="Call John at 555-1234")
        result = pii_metric(gold, pred)
        assert 0 < result.score < 1
        assert "Incorrect" in result.feedback
        assert "Expected" in result.feedback

    def test_strips_whitespace(self):
        gold = dspy.Example(redacted_text="Call [GIVENNAME1]")
        pred = dspy.Prediction(redacted_text="  Call [GIVENNAME1]  ")
        result = pii_metric(gold, pred)
        assert result.score == 1

    def test_accepts_optional_params(self):
        gold = dspy.Example(redacted_text="text")
        pred = dspy.Prediction(redacted_text="text")
        result = pii_metric(gold, pred, trace="t", pred_name="p", pred_trace="pt")
        assert result.score == 1


class TestPrepareExamples:
    def _make_dataset(self, n):
        rows = [
            {"source_text": f"text {i}", "target_text": f"redacted {i}"}
            for i in range(n)
        ]
        mock_ds = MagicMock()
        mock_ds.select.return_value = rows
        mock_ds.__len__ = lambda self: n
        return mock_ds

    def test_splits_by_train_val_size(self):
        ds = self._make_dataset(100)
        train, val = prepare_examples(ds, train_size=90, val_size=10)
        assert len(train) == 90
        assert len(val) == 10

    def test_creates_dspy_examples(self):
        ds = self._make_dataset(10)
        train, val = prepare_examples(ds, train_size=7, val_size=3)
        ex = train[0]
        assert hasattr(ex, "text")
        assert hasattr(ex, "redacted_text")
        assert "text" in ex.inputs()

    def test_caps_at_dataset_size(self):
        ds = self._make_dataset(5)
        train, val = prepare_examples(ds, train_size=450, val_size=50)
        assert len(train) + len(val) == 5

    def test_reads_env_defaults(self, monkeypatch):
        monkeypatch.setenv("OPTIMIZE_TRAIN_SIZE", "80")
        monkeypatch.setenv("OPTIMIZE_VAL_SIZE", "20")
        ds = self._make_dataset(100)
        train, val = prepare_examples(ds)
        assert len(train) == 80
        assert len(val) == 20


class TestPrepareEvalExamples:
    def _make_dataset(self, n):
        rows = [
            {"source_text": f"text {i}", "target_text": f"redacted {i}"}
            for i in range(n)
        ]
        mock_ds = MagicMock()
        mock_ds.select.return_value = rows
        mock_ds.__len__ = lambda self: n
        return mock_ds

    def test_picks_from_offset(self):
        ds = self._make_dataset(1000)
        prepare_eval_examples(ds, eval_size=100, offset=500)
        ds.select.assert_called_once_with(range(500, 600))

    def test_default_size_and_offset(self):
        ds = self._make_dataset(2000)
        prepare_eval_examples(ds, eval_size=500, offset=500)
        ds.select.assert_called_once_with(range(500, 1000))

    def test_creates_dspy_examples(self):
        ds = self._make_dataset(1000)
        examples = prepare_eval_examples(ds, eval_size=10, offset=0)
        ex = examples[0]
        assert hasattr(ex, "text")
        assert hasattr(ex, "redacted_text")
        assert "text" in ex.inputs()

    def test_caps_at_dataset_end(self):
        ds = self._make_dataset(550)
        prepare_eval_examples(ds, eval_size=500, offset=500)
        ds.select.assert_called_once_with(range(500, 550))

    def test_reads_env_defaults(self, monkeypatch):
        monkeypatch.setenv("EVALUATE_SIZE", "200")
        monkeypatch.setenv("EVALUATE_OFFSET", "100")
        ds = self._make_dataset(1000)
        prepare_eval_examples(ds)
        ds.select.assert_called_once_with(range(100, 300))

    def test_offset_defaults_to_optimize_sizes(self, monkeypatch):
        monkeypatch.setenv("OPTIMIZE_TRAIN_SIZE", "800")
        monkeypatch.setenv("OPTIMIZE_VAL_SIZE", "200")
        monkeypatch.delenv("EVALUATE_OFFSET", raising=False)
        ds = self._make_dataset(2000)
        prepare_eval_examples(ds, eval_size=100)
        ds.select.assert_called_once_with(range(1000, 1100))


class TestLoadOptimizedModel:
    def test_returns_none_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "optimizer.OPTIMIZED_MODEL_PATH", str(tmp_path / "nope.json")
        )
        assert load_optimized_model() is None
