import subprocess
import sys
from unittest.mock import MagicMock, patch


from main import redact


class TestRedactLogging:
    """Test that redact() emits the expected log messages."""

    @patch("main.PIIRedactor")
    @patch("main.dspy")
    def test_logs_model_at_info(self, mock_dspy, mock_redactor_cls, caplog):
        mock_result = MagicMock()
        mock_result.redacted_text = "redacted"
        mock_result.entities = []
        mock_redactor_cls.return_value.return_value = mock_result

        with caplog.at_level("INFO", logger="main"):
            redact("hello")

        assert any("Using model:" in r.message for r in caplog.records)

    @patch("main.PIIRedactor")
    @patch("main.dspy")
    def test_logs_input_at_debug(self, mock_dspy, mock_redactor_cls, caplog):
        mock_result = MagicMock()
        mock_result.redacted_text = "redacted"
        mock_result.entities = []
        mock_redactor_cls.return_value.return_value = mock_result

        with caplog.at_level("DEBUG", logger="main"):
            redact("secret text")

        assert any("secret text" in r.message for r in caplog.records)

    @patch("main.PIIRedactor")
    @patch("main.dspy")
    def test_logs_entities_at_info(self, mock_dspy, mock_redactor_cls, caplog):
        mock_result = MagicMock()
        mock_result.redacted_text = "redacted"
        mock_result.entities = [{"value": "John", "label": "GIVENNAME1"}]
        mock_redactor_cls.return_value.return_value = mock_result

        with caplog.at_level("INFO", logger="main"):
            redact("Call John")

        assert any("Entities found:" in r.message for r in caplog.records)

    @patch("main.PIIRedactor")
    @patch("main.dspy")
    def test_logs_redacted_text_at_debug(self, mock_dspy, mock_redactor_cls, caplog):
        mock_result = MagicMock()
        mock_result.redacted_text = "[GIVENNAME1]"
        mock_result.entities = []
        mock_redactor_cls.return_value.return_value = mock_result

        with caplog.at_level("DEBUG", logger="main"):
            redact("John")

        assert any("[GIVENNAME1]" in r.message for r in caplog.records)

    @patch("main.PIIRedactor")
    @patch("main.dspy")
    def test_no_debug_logs_at_info_level(self, mock_dspy, mock_redactor_cls, caplog):
        mock_result = MagicMock()
        mock_result.redacted_text = "redacted"
        mock_result.entities = []
        mock_redactor_cls.return_value.return_value = mock_result

        with caplog.at_level("INFO", logger="main"):
            redact("hello")

        debug_records = [r for r in caplog.records if r.levelname == "DEBUG"]
        assert len(debug_records) == 0


class TestCLIFlags:
    """Test that CLI flags produce expected output."""

    def test_default_output(self):
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert result.stdout.strip()

    def test_verbose_shows_dspy_history(self):
        result = subprocess.run(
            [sys.executable, "main.py", "-v", "Call John at 555-1234"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert "--- DSPy History ---" in result.stdout

    def test_debug_shows_info_logs(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--debug", "Call John at 555-1234"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert "Using model:" in result.stderr
