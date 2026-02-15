from unittest.mock import MagicMock

import dspy

from optimizer import (
    extract_pii_labels,
    load_optimized_model,
    pii_metric,
    pii_only_f1,
    prepare_examples,
)


class TestExtractPiiLabels:
    def test_extracts_labels(self):
        text = "Call [GIVENNAME1] [LASTNAME1] at [TEL]."
        assert extract_pii_labels(text) == ["GIVENNAME1", "LASTNAME1", "TEL"]

    def test_empty_text(self):
        assert extract_pii_labels("No PII here.") == []

    def test_handles_numbered_labels(self):
        text = "[GIVENNAME1] and [GIVENNAME2]"
        assert extract_pii_labels(text) == ["GIVENNAME1", "GIVENNAME2"]


class TestPiiOnlyF1:
    def test_perfect_match(self):
        p, r, f1 = pii_only_f1(
            "Call [GIVENNAME1] at [TEL].",
            "Call [GIVENNAME1] at [TEL].",
        )
        assert f1 == 1.0
        assert p == 1.0
        assert r == 1.0

    def test_no_pii_in_either(self):
        p, r, f1 = pii_only_f1("Hello world.", "Hello world.")
        assert f1 == 1.0

    def test_missing_label_hurts_recall(self):
        p, r, f1 = pii_only_f1(
            "Call [GIVENNAME1] at [TEL].",
            "Call [GIVENNAME1] at 555-1234.",
        )
        assert p == 1.0
        assert r == 0.5
        assert 0 < f1 < 1

    def test_extra_label_hurts_precision(self):
        p, r, f1 = pii_only_f1(
            "Call [GIVENNAME1] at home.",
            "Call [GIVENNAME1] at [TEL].",
        )
        assert p == 0.5
        assert r == 1.0
        assert 0 < f1 < 1

    def test_completely_wrong(self):
        p, r, f1 = pii_only_f1(
            "Call [GIVENNAME1] at [TEL].",
            "Call John at 555-1234.",
        )
        assert f1 == 0.0

    def test_gold_has_pii_pred_does_not(self):
        p, r, f1 = pii_only_f1("[TEL]", "555-1234")
        assert f1 == 0.0

    def test_pred_has_pii_gold_does_not(self):
        p, r, f1 = pii_only_f1("555-1234", "[TEL]")
        assert f1 == 0.0

    def test_duplicate_labels(self):
        p, r, f1 = pii_only_f1(
            "[TEL] and [TEL]",
            "[TEL]",
        )
        assert p == 1.0
        assert r == 0.5


class TestPiiMetric:
    def test_exact_match_scores_one(self):
        gold = dspy.Example(redacted_text="Call [GIVENNAME1] at [TEL]")
        pred = dspy.Prediction(redacted_text="Call [GIVENNAME1] at [TEL]")
        result = pii_metric(gold, pred)
        assert result.score == 1
        assert "Correct" in result.feedback

    def test_missed_label_feedback(self):
        gold = dspy.Example(redacted_text="Call [GIVENNAME1] at [TEL]")
        pred = dspy.Prediction(redacted_text="Call [GIVENNAME1] at 555-1234")
        result = pii_metric(gold, pred)
        assert 0 < result.score < 1
        assert "Missed PII labels" in result.feedback
        assert "TEL" in result.feedback

    def test_extra_label_feedback(self):
        gold = dspy.Example(redacted_text="Call [GIVENNAME1] at home")
        pred = dspy.Prediction(redacted_text="Call [GIVENNAME1] at [TEL]")
        result = pii_metric(gold, pred)
        assert 0 < result.score < 1
        assert "Extra PII labels" in result.feedback
        assert "TEL" in result.feedback

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

    def test_includes_expected_and_got(self):
        gold = dspy.Example(redacted_text="[GIVENNAME1] [LASTNAME1]")
        pred = dspy.Prediction(redacted_text="[GIVENNAME1] Smith")
        result = pii_metric(gold, pred)
        assert "Expected" in result.feedback
        assert "Got" in result.feedback

    def test_includes_precision_recall(self):
        gold = dspy.Example(redacted_text="[GIVENNAME1] [LASTNAME1]")
        pred = dspy.Prediction(redacted_text="[GIVENNAME1] Smith")
        result = pii_metric(gold, pred)
        assert "precision=" in result.feedback
        assert "recall=" in result.feedback

    def test_includes_text_f1_reference(self):
        gold = dspy.Example(redacted_text="[GIVENNAME1] [LASTNAME1]")
        pred = dspy.Prediction(redacted_text="[GIVENNAME1] Smith")
        result = pii_metric(gold, pred)
        assert "Full-text token F1=" in result.feedback

    def test_no_pii_text_scores_one(self):
        gold = dspy.Example(redacted_text="No PII here.")
        pred = dspy.Prediction(redacted_text="No PII here.")
        result = pii_metric(gold, pred)
        assert result.score == 1.0


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


class TestLoadOptimizedModel:
    def test_returns_none_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "optimizer.OPTIMIZED_MODEL_PATH", str(tmp_path / "nope.json")
        )
        assert load_optimized_model() is None
