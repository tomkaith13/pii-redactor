import pytest

from main import PIIEntity


class TestPIIEntity:
    def test_create(self):
        e = PIIEntity(value="John", label="GIVENNAME1")
        assert e.value == "John"
        assert e.label == "GIVENNAME1"

    def test_rejects_missing_fields(self):
        with pytest.raises(Exception):
            PIIEntity(value="John")
