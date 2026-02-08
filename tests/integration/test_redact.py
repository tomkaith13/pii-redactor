import pytest

from main import redact

pytestmark = pytest.mark.integration


class TestRedactIntegration:
    def test_name_and_phone(self):
        result = redact("Call John Smith at 555-123-4567")
        assert "[GIVENNAME1]" in result or "[GIVENNAME" in result
        assert "[LASTNAME1]" in result or "[LASTNAME" in result
        assert "[TEL]" in result
        assert "John" not in result
        assert "Smith" not in result
        assert "555-123-4567" not in result

    def test_email(self):
        result = redact("Email me at alice@example.com")
        assert "[EMAIL]" in result
        assert "alice@example.com" not in result

    def test_ssn(self):
        result = redact("My SSN is 123-45-6789")
        assert "[SOCIALNUMBER]" in result
        assert "123-45-6789" not in result

    def test_address(self):
        result = redact("I live at 42 Oak Street, Portland, OR 97201")
        assert "[STREET]" in result
        assert "[CITY]" in result
        assert "[STATE]" in result
        assert "[POSTCODE]" in result

    def test_no_pii(self):
        text = "The weather is nice today."
        result = redact(text)
        assert "[" not in result or result == text

    def test_multiple_people(self):
        result = redact("Alice Brown met Bob Davis at the cafe.")
        assert "Alice" not in result
        assert "Brown" not in result
        assert "Bob" not in result
        assert "Davis" not in result

    def test_ip_address(self):
        result = redact("Server at 10.0.0.1 is down")
        assert "[IP]" in result
        assert "10.0.0.1" not in result

    def test_returns_string(self):
        result = redact("Hi there")
        assert isinstance(result, str)
