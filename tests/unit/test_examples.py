import re

from examples import EXAMPLES

VALID_LABELS = {
    "GIVENNAME1",
    "GIVENNAME2",
    "LASTNAME1",
    "LASTNAME2",
    "LASTNAME3",
    "TITLE",
    "TEL",
    "EMAIL",
    "USERNAME",
    "SOCIALNUMBER",
    "IDCARD",
    "DRIVERLICENSE",
    "PASSPORT",
    "STREET",
    "BUILDING",
    "CITY",
    "STATE",
    "POSTCODE",
    "COUNTRY",
    "SECADDRESS",
    "GEOCOORD",
    "SEX",
    "BOD",
    "PASS",
    "IP",
    "DATE",
    "TIME",
}

LABEL_PATTERN = re.compile(r"\[([A-Z0-9]+)\]")


class TestExamplesStructure:
    def test_minimum_count(self):
        assert len(EXAMPLES) >= 20

    def test_all_have_required_fields(self):
        for i, ex in enumerate(EXAMPLES):
            assert hasattr(ex, "text"), f"Example {i} missing 'text'"
            assert hasattr(ex, "entities"), f"Example {i} missing 'entities'"
            assert hasattr(ex, "redacted_text"), f"Example {i} missing 'redacted_text'"

    def test_text_is_input(self):
        for i, ex in enumerate(EXAMPLES):
            assert "text" in ex.inputs(), f"Example {i}: 'text' not marked as input"

    def test_entities_use_valid_labels(self):
        for i, ex in enumerate(EXAMPLES):
            for ent in ex.entities:
                assert ent["label"] in VALID_LABELS, (
                    f"Example {i}: unknown label '{ent['label']}'"
                )

    def test_entity_values_appear_in_text(self):
        for i, ex in enumerate(EXAMPLES):
            for ent in ex.entities:
                assert ent["value"] in ex.text, (
                    f"Example {i}: entity value '{ent['value']}' not in text"
                )

    def test_redacted_text_contains_labels(self):
        for i, ex in enumerate(EXAMPLES):
            labels_in_redacted = set(LABEL_PATTERN.findall(ex.redacted_text))
            entity_labels = {ent["label"] for ent in ex.entities}
            assert entity_labels == labels_in_redacted, (
                f"Example {i}: label mismatch — "
                f"entities={entity_labels}, redacted={labels_in_redacted}"
            )

    def test_redacted_text_has_no_raw_pii(self):
        for i, ex in enumerate(EXAMPLES):
            for ent in ex.entities:
                # Use word-boundary regex to avoid false positives
                # (e.g. "CO" matching inside "[POSTCODE]")
                pattern = re.compile(r"\b" + re.escape(ent["value"]) + r"\b")
                assert not pattern.search(ex.redacted_text), (
                    f"Example {i}: raw PII '{ent['value']}' still in redacted_text"
                )


class TestLabelCoverage:
    """Ensure examples collectively cover every label at least once."""

    def _all_used_labels(self):
        labels = set()
        for ex in EXAMPLES:
            for ent in ex.entities:
                labels.add(ent["label"])
        return labels

    def test_all_labels_covered(self):
        used = self._all_used_labels()
        missing = VALID_LABELS - used
        assert not missing, f"Labels not covered by any example: {missing}"
