from unittest.mock import MagicMock


from evaluator import prepare_eval_examples


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


class TestPrepareEvalExamplesRandomize:
    def _make_dataset(self, n):
        rows = [
            {"source_text": f"text {i}", "target_text": f"redacted {i}"}
            for i in range(n)
        ]
        mock_ds = MagicMock()
        mock_ds.select.return_value = rows
        mock_ds.__len__ = lambda self: n
        return mock_ds

    def test_randomize_selects_correct_count(self, monkeypatch):
        monkeypatch.setenv("OPTIMIZE_TRAIN_SIZE", "100")
        monkeypatch.setenv("OPTIMIZE_VAL_SIZE", "50")
        monkeypatch.delenv("EVALUATE_SEED", raising=False)
        ds = self._make_dataset(1000)
        prepare_eval_examples(ds, eval_size=50, randomize=True)
        indices = ds.select.call_args[0][0]
        assert len(indices) == 50

    def test_randomize_excludes_optimization_indices(self, monkeypatch):
        monkeypatch.setenv("OPTIMIZE_TRAIN_SIZE", "200")
        monkeypatch.setenv("OPTIMIZE_VAL_SIZE", "100")
        monkeypatch.delenv("EVALUATE_SEED", raising=False)
        ds = self._make_dataset(1000)
        prepare_eval_examples(ds, eval_size=100, randomize=True)
        indices = ds.select.call_args[0][0]
        assert all(i >= 300 for i in indices)

    def test_randomize_indices_are_sorted(self, monkeypatch):
        monkeypatch.setenv("OPTIMIZE_TRAIN_SIZE", "100")
        monkeypatch.setenv("OPTIMIZE_VAL_SIZE", "50")
        monkeypatch.delenv("EVALUATE_SEED", raising=False)
        ds = self._make_dataset(1000)
        prepare_eval_examples(ds, eval_size=200, randomize=True)
        indices = ds.select.call_args[0][0]
        assert indices == sorted(indices)

    def test_randomize_seed_reproducible(self, monkeypatch):
        monkeypatch.setenv("OPTIMIZE_TRAIN_SIZE", "100")
        monkeypatch.setenv("OPTIMIZE_VAL_SIZE", "50")
        monkeypatch.setenv("EVALUATE_SEED", "42")
        ds1 = self._make_dataset(1000)
        prepare_eval_examples(ds1, eval_size=50, randomize=True)
        indices1 = ds1.select.call_args[0][0]

        ds2 = self._make_dataset(1000)
        prepare_eval_examples(ds2, eval_size=50, randomize=True)
        indices2 = ds2.select.call_args[0][0]

        assert indices1 == indices2

    def test_randomize_caps_at_pool_size(self, monkeypatch):
        monkeypatch.setenv("OPTIMIZE_TRAIN_SIZE", "80")
        monkeypatch.setenv("OPTIMIZE_VAL_SIZE", "20")
        monkeypatch.delenv("EVALUATE_SEED", raising=False)
        ds = self._make_dataset(120)
        # Pool is 120-100=20, but we ask for 50
        prepare_eval_examples(ds, eval_size=50, randomize=True)
        indices = ds.select.call_args[0][0]
        assert len(indices) == 20
