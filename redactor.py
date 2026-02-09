import dspy
from pydantic import BaseModel


class PIIEntity(BaseModel):
    value: str
    label: str


class IdentifyPII(dspy.Signature):
    """Identify all PII entities in the text and produce a redacted version.

    Use these labels (from ai4privacy/pii-masking-300k):
    Names: GIVENNAME1, GIVENNAME2, LASTNAME1, LASTNAME2, TITLE
    Contact: TEL, EMAIL, USERNAME
    IDs: SOCIALNUMBER, IDCARD, DRIVERLICENSE, PASSPORT
    Location: STREET, BUILDING, CITY, STATE, POSTCODE, COUNTRY, SECADDRESS
    Personal: SEX, BOD, PASS
    Digital: IP
    Time: DATE, TIME
    """

    text: str = dspy.InputField(desc="Text that may contain PII")
    entities: list[PIIEntity] = dspy.OutputField(
        desc="All PII entities found with their labels"
    )
    redacted_text: str = dspy.OutputField(
        desc="Text with each PII value replaced by [LABEL]",
    )


class PIIRedactor(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        from examples import EXAMPLES

        self.cot = dspy.ChainOfThought(IdentifyPII)
        self.cot.demos = EXAMPLES

    def forward(self, text: str) -> dspy.Prediction:
        return self.cot(text=text)
