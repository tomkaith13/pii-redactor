import dspy

EXAMPLES = [
    # 1. Names: GIVENNAME1, LASTNAME1
    dspy.Example(
        text="Call John Smith at the office.",
        entities=[
            {"value": "John", "label": "GIVENNAME1"},
            {"value": "Smith", "label": "LASTNAME1"},
        ],
        redacted_text="Call [GIVENNAME1] [LASTNAME1] at the office.",
    ).with_inputs("text"),
    # 2. Names: TITLE, GIVENNAME1, LASTNAME1
    dspy.Example(
        text="Dr. Emily Watson will see you now.",
        entities=[
            {"value": "Dr.", "label": "TITLE"},
            {"value": "Emily", "label": "GIVENNAME1"},
            {"value": "Watson", "label": "LASTNAME1"},
        ],
        redacted_text="[TITLE] [GIVENNAME1] [LASTNAME1] will see you now.",
    ).with_inputs("text"),
    # 3. TEL
    dspy.Example(
        text="Reach me at 555-123-4567 for details.",
        entities=[
            {"value": "555-123-4567", "label": "TEL"},
        ],
        redacted_text="Reach me at [TEL] for details.",
    ).with_inputs("text"),
    # 4. EMAIL
    dspy.Example(
        text="Send the report to jane.doe@example.com by Friday.",
        entities=[
            {"value": "jane.doe@example.com", "label": "EMAIL"},
        ],
        redacted_text="Send the report to [EMAIL] by Friday.",
    ).with_inputs("text"),
    # 5. USERNAME
    dspy.Example(
        text="Follow @cooluser99 on social media.",
        entities=[
            {"value": "@cooluser99", "label": "USERNAME"},
        ],
        redacted_text="Follow [USERNAME] on social media.",
    ).with_inputs("text"),
    # 6. SOCIALNUMBER
    dspy.Example(
        text="My social security number is 123-45-6789.",
        entities=[
            {"value": "123-45-6789", "label": "SOCIALNUMBER"},
        ],
        redacted_text="My social security number is [SOCIALNUMBER].",
    ).with_inputs("text"),
    # 7. IDCARD
    dspy.Example(
        text="Her national ID card number is A1234567.",
        entities=[
            {"value": "A1234567", "label": "IDCARD"},
        ],
        redacted_text="Her national ID card number is [IDCARD].",
    ).with_inputs("text"),
    # 8. DRIVERLICENSE
    dspy.Example(
        text="His driver's license number is D400-123-45-678.",
        entities=[
            {"value": "D400-123-45-678", "label": "DRIVERLICENSE"},
        ],
        redacted_text="His driver's license number is [DRIVERLICENSE].",
    ).with_inputs("text"),
    # 9. PASSPORT
    dspy.Example(
        text="Please provide your passport number: X12345678.",
        entities=[
            {"value": "X12345678", "label": "PASSPORT"},
        ],
        redacted_text="Please provide your passport number: [PASSPORT].",
    ).with_inputs("text"),
    # 10. STREET, CITY, STATE, POSTCODE
    dspy.Example(
        text="Ship to 742 Evergreen Terrace, Springfield, IL 62704.",
        entities=[
            {"value": "742 Evergreen Terrace", "label": "STREET"},
            {"value": "Springfield", "label": "CITY"},
            {"value": "IL", "label": "STATE"},
            {"value": "62704", "label": "POSTCODE"},
        ],
        redacted_text="Ship to [STREET], [CITY], [STATE] [POSTCODE].",
    ).with_inputs("text"),
    # 11. BUILDING
    dspy.Example(
        text="The meeting is at Chrysler Building, floor 12.",
        entities=[
            {"value": "Chrysler Building", "label": "BUILDING"},
        ],
        redacted_text="The meeting is at [BUILDING], floor 12.",
    ).with_inputs("text"),
    # 12. COUNTRY
    dspy.Example(
        text="She moved to Germany last year.",
        entities=[
            {"value": "Germany", "label": "COUNTRY"},
        ],
        redacted_text="She moved to [COUNTRY] last year.",
    ).with_inputs("text"),
    # 13. SECADDRESS
    dspy.Example(
        text="Deliver to Apt 4B, second entrance.",
        entities=[
            {"value": "Apt 4B", "label": "SECADDRESS"},
        ],
        redacted_text="Deliver to [SECADDRESS], second entrance.",
    ).with_inputs("text"),
    # 14. SEX
    dspy.Example(
        text="The patient is male, age 34.",
        entities=[
            {"value": "male", "label": "SEX"},
        ],
        redacted_text="The patient is [SEX], age 34.",
    ).with_inputs("text"),
    # 15. BOD (date of birth)
    dspy.Example(
        text="Date of birth: 03/15/1990.",
        entities=[
            {"value": "03/15/1990", "label": "BOD"},
        ],
        redacted_text="Date of birth: [BOD].",
    ).with_inputs("text"),
    # 16. PASS (password)
    dspy.Example(
        text="Your temporary password is Xk9#mP2!qR.",
        entities=[
            {"value": "Xk9#mP2!qR", "label": "PASS"},
        ],
        redacted_text="Your temporary password is [PASS].",
    ).with_inputs("text"),
    # 17. IP
    dspy.Example(
        text="The server IP is 192.168.1.100.",
        entities=[
            {"value": "192.168.1.100", "label": "IP"},
        ],
        redacted_text="The server IP is [IP].",
    ).with_inputs("text"),
    # 18. DATE
    dspy.Example(
        text="The contract expires on January 15, 2025.",
        entities=[
            {"value": "January 15, 2025", "label": "DATE"},
        ],
        redacted_text="The contract expires on [DATE].",
    ).with_inputs("text"),
    # 19. TIME
    dspy.Example(
        text="The appointment is at 3:30 PM.",
        entities=[
            {"value": "3:30 PM", "label": "TIME"},
        ],
        redacted_text="The appointment is at [TIME].",
    ).with_inputs("text"),
    # 20. GIVENNAME2, LASTNAME2 (second person)
    dspy.Example(
        text="Alice Johnson and Bob Williams attended the meeting.",
        entities=[
            {"value": "Alice", "label": "GIVENNAME1"},
            {"value": "Johnson", "label": "LASTNAME1"},
            {"value": "Bob", "label": "GIVENNAME2"},
            {"value": "Williams", "label": "LASTNAME2"},
        ],
        redacted_text=(
            "[GIVENNAME1] [LASTNAME1] and [GIVENNAME2] [LASTNAME2]"
            " attended the meeting."
        ),
    ).with_inputs("text"),
    # 21. Multi-entity: name + email + tel
    dspy.Example(
        text="Contact Sarah Miller at sarah.m@corp.io or 408-555-0199.",
        entities=[
            {"value": "Sarah", "label": "GIVENNAME1"},
            {"value": "Miller", "label": "LASTNAME1"},
            {"value": "sarah.m@corp.io", "label": "EMAIL"},
            {"value": "408-555-0199", "label": "TEL"},
        ],
        redacted_text=("Contact [GIVENNAME1] [LASTNAME1] at [EMAIL] or [TEL]."),
    ).with_inputs("text"),
    # 22. Multi-entity: address + date + time
    dspy.Example(
        text="Visit us at 100 Main St, Denver, CO 80202 on March 3, 2025 at 10:00 AM.",
        entities=[
            {"value": "100 Main St", "label": "STREET"},
            {"value": "Denver", "label": "CITY"},
            {"value": "CO", "label": "STATE"},
            {"value": "80202", "label": "POSTCODE"},
            {"value": "March 3, 2025", "label": "DATE"},
            {"value": "10:00 AM", "label": "TIME"},
        ],
        redacted_text=(
            "Visit us at [STREET], [CITY], [STATE] [POSTCODE] on [DATE] at [TIME]."
        ),
    ).with_inputs("text"),
    # 23. Multi-entity: IDs + name
    dspy.Example(
        text="Mr. Carlos Rivera, SSN 987-65-4321, license CA-98765432.",
        entities=[
            {"value": "Mr.", "label": "TITLE"},
            {"value": "Carlos", "label": "GIVENNAME1"},
            {"value": "Rivera", "label": "LASTNAME1"},
            {"value": "987-65-4321", "label": "SOCIALNUMBER"},
            {"value": "CA-98765432", "label": "DRIVERLICENSE"},
        ],
        redacted_text=(
            "[TITLE] [GIVENNAME1] [LASTNAME1], SSN [SOCIALNUMBER],"
            " license [DRIVERLICENSE]."
        ),
    ).with_inputs("text"),
    # 24. Multi-entity: personal info
    dspy.Example(
        text="Patient: female, born 11/22/1985, password reset to Abc!2345.",
        entities=[
            {"value": "female", "label": "SEX"},
            {"value": "11/22/1985", "label": "BOD"},
            {"value": "Abc!2345", "label": "PASS"},
        ],
        redacted_text=("Patient: [SEX], born [BOD], password reset to [PASS]."),
    ).with_inputs("text"),
    # 25. Multi-entity: digital + location + name
    dspy.Example(
        text="User @netadmin from IP 10.0.0.1 accessed the server in Tower A, Tokyo, Japan.",
        entities=[
            {"value": "@netadmin", "label": "USERNAME"},
            {"value": "10.0.0.1", "label": "IP"},
            {"value": "Tower A", "label": "BUILDING"},
            {"value": "Tokyo", "label": "CITY"},
            {"value": "Japan", "label": "COUNTRY"},
        ],
        redacted_text=(
            "User [USERNAME] from IP [IP] accessed the server"
            " in [BUILDING], [CITY], [COUNTRY]."
        ),
    ).with_inputs("text"),
    # 26. Complex multi-entity: name, email, tel, address, SSN, DOB, ID, bank details
    dspy.Example(
        text=(
            "Please contact Dr. Sarah Johnson at sarah.johnson@medcorp.com"
            " or call her at (415) 555-0198. She lives at 742 Evergreen Terrace,"
            " Springfield, IL 62704. Her SSN is 123-45-6789, date of birth is"
            " March 15, 1985, and her employee ID is EMP-29481. Send payment"
            " of $4,500 to account number 8827364510 at First National Bank,"
            " routing number 071000013."
        ),
        entities=[
            {"value": "Dr.", "label": "TITLE"},
            {"value": "Sarah", "label": "GIVENNAME1"},
            {"value": "Johnson", "label": "LASTNAME1"},
            {"value": "sarah.johnson@medcorp.com", "label": "EMAIL"},
            {"value": "(415) 555-0198", "label": "TEL"},
            {"value": "742 Evergreen Terrace", "label": "STREET"},
            {"value": "Springfield", "label": "CITY"},
            {"value": "IL", "label": "STATE"},
            {"value": "62704", "label": "POSTCODE"},
            {"value": "123-45-6789", "label": "SOCIALNUMBER"},
            {"value": "March 15, 1985", "label": "BOD"},
            {"value": "EMP-29481", "label": "IDCARD"},
        ],
        redacted_text=(
            "Please contact [TITLE] [GIVENNAME1] [LASTNAME1] at [EMAIL]"
            " or call her at [TEL]. She lives at [STREET],"
            " [CITY], [STATE] [POSTCODE]. Her SSN is [SOCIALNUMBER], date of birth is"
            " [BOD], and her employee ID is [IDCARD]. Send payment"
            " of $4,500 to account number 8827364510 at First National Bank,"
            " routing number 071000013."
        ),
    ).with_inputs("text"),
]
