from main import PIIRedactor


class TestPIIRedactor:
    def test_has_cot_predictor(self):
        r = PIIRedactor()
        assert hasattr(r, "cot")
