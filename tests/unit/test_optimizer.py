from unittest.mock import MagicMock

import dspy

from optimizer import (
    extract_pii_labels,
    hybrid_pii_score,
    load_optimized_model,
    pii_metric,
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


class TestHybridPiiScore:
    def test_perfect_match(self):
        det, cls, score, n = hybrid_pii_score(
            "Call [GIVENNAME1] at [TEL].",
            "Call [GIVENNAME1] at [TEL].",
        )
        assert det == 1.0
        assert cls == 1.0
        assert score == 1.0
        assert n == 2

    def test_no_pii_in_either(self):
        det, cls, score, n = hybrid_pii_score("Hello world.", "Hello world.")
        assert score == 1.0
        assert n == 0

    def test_missing_label_hurts_detection_recall(self):
        det, cls, score, n = hybrid_pii_score(
            "Call [GIVENNAME1] at [TEL].",
            "Call [GIVENNAME1] at 555-1234.",
        )
        assert det == 0.5
        assert cls == 1.0
        assert score == 0.75 * 0.5 + 0.25 * 1.0

    def test_over_redaction_does_not_hurt_detection(self):
        det, cls, score, n = hybrid_pii_score(
            "Call [GIVENNAME1] at home.",
            "Call [GIVENNAME1] at [TEL].",
        )
        assert det == 1.0  # over-redaction is acceptable

    def test_wrong_label_only_hurts_classification(self):
        det, cls, score, n = hybrid_pii_score(
            "ID: [PASSPORT]",
            "ID: [IDCARD]",
        )
        assert det == 1.0
        assert cls == 0.0
        assert score == 0.75 * 1.0 + 0.25 * 0.0

    def test_completely_wrong(self):
        det, cls, score, n = hybrid_pii_score(
            "Call [GIVENNAME1] at [TEL].",
            "Call John at 555-1234.",
        )
        assert score == 0.0

    def test_gold_has_pii_pred_does_not(self):
        det, cls, score, n = hybrid_pii_score("[TEL]", "555-1234")
        assert score == 0.0

    def test_pred_has_pii_gold_does_not(self):
        det, cls, score, n = hybrid_pii_score("555-1234", "[TEL]")
        assert score == 1.0  # over-redaction is acceptable

    def test_duplicate_labels_under_redacted(self):
        det, cls, score, n = hybrid_pii_score(
            "[TEL] and [TEL]",
            "[TEL]",
        )
        assert det == 0.5
        assert cls == 1.0

    def test_hybrid_weights(self):
        """Score = 0.75 * detection_recall + 0.25 * classification_accuracy."""
        det, cls, score, n = hybrid_pii_score(
            "[GIVENNAME1] [LASTNAME1] [TEL]",
            "[GIVENNAME1] [IDCARD] [EMAIL]",
        )
        assert det == 1.0  # 3 gold, 3 pred
        assert n == 1  # only GIVENNAME1 matches
        assert cls == 1 / 3
        expected = 0.75 * 1.0 + 0.25 * (1 / 3)
        assert abs(score - expected) < 1e-9

    def test_mixed_under_over_and_wrong_labels(self):
        """Combines under-redaction, over-redaction, and wrong labels."""
        det, cls, score, n = hybrid_pii_score(
            "[GIVENNAME1] [LASTNAME1] [TEL] [EMAIL]",
            "[GIVENNAME1] [IDCARD] [TEL] [IP] [USERNAME]",
        )
        assert det == 1.0  # min(4, 5) / 4 = 1.0
        assert n == 2  # GIVENNAME1 and TEL match
        assert cls == 0.5  # 2 / 4 detected
        expected = 0.75 * 1.0 + 0.25 * 0.5
        assert abs(score - expected) < 1e-9


class TestPiiMetric:
    def test_exact_match_scores_one(self):
        gold = dspy.Example(redacted_text="Call [GIVENNAME1] at [TEL]")
        pred = dspy.Prediction(redacted_text="Call [GIVENNAME1] at [TEL]")
        result = pii_metric(gold, pred)
        assert result.score == 1
        assert "Correct" in result.feedback

    def test_missed_pii_critical_feedback(self):
        gold = dspy.Example(redacted_text="Call [GIVENNAME1] at [TEL]")
        pred = dspy.Prediction(redacted_text="Call [GIVENNAME1] at 555-1234")
        result = pii_metric(gold, pred)
        assert 0 < result.score < 1
        assert "CRITICAL" in result.feedback

    def test_wrong_label_minor_feedback(self):
        gold = dspy.Example(redacted_text="ID: [PASSPORT]")
        pred = dspy.Prediction(redacted_text="ID: [IDCARD]")
        result = pii_metric(gold, pred)
        assert result.score == 0.75  # detection perfect, classification zero
        assert "Minor:" in result.feedback
        assert "Missing labels: PASSPORT" in result.feedback
        assert "Unexpected labels: IDCARD" in result.feedback

    def test_over_redaction_note_feedback(self):
        gold = dspy.Example(redacted_text="Call [GIVENNAME1] at home")
        pred = dspy.Prediction(redacted_text="Call [GIVENNAME1] at [TEL]")
        result = pii_metric(gold, pred)
        assert result.score == 1.0  # over-redaction not penalised
        assert "Note:" in result.feedback
        assert "Over-redacted" in result.feedback

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

    def test_includes_detection_and_classification(self):
        gold = dspy.Example(redacted_text="[GIVENNAME1] [LASTNAME1]")
        pred = dspy.Prediction(redacted_text="[GIVENNAME1] Smith")
        result = pii_metric(gold, pred)
        assert "detection_recall=" in result.feedback
        assert "classification_acc=" in result.feedback

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
