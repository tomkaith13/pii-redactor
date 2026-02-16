import dspy

# Row IDs from ai4privacy/pii-masking-300k used as few-shot demos.
# These are excluded from optimize/eval sets to prevent data leakage.
FEWSHOT_ROW_IDS: set[str] = {
    "40767A",
    "40768A",
    "40768B",
    "40768C",
    "40769A",
    "40769B",
    "40769C",
    "40772A",
    "40773",
    "40774A",
    "40774B",
    "40774C",
    "40775A",
    "40775B",
    "40775C",
    "40777A",
    "40777D",
    "40778B",
    "40781A",
    "40781B",
    "40782A",
    "40782C",
    "40976C",
    "41153A",
    "51236",
}

EXAMPLES: list[dspy.Example] = [
    # 1. TIME, USERNAME
    dspy.Example(
        text="Subject: Group Messaging for Admissions Process\n\nGood morning, everyone,\n\nI hope this message finds you well. As we continue our admissions processes, I would like to update you on the latest developments and key information. Please find below the timeline for our upcoming meetings:\n\n- wynqvrh053 - Meeting at 10:20am\n- luka.burg - Meeting at 21\n- qahil.wittauer - Meeting at quarter past 13\n- gholamhossein.ruschke - Meeting at 9:47 PM\n- pdmjrsyoz1460 ",
        entities=[
            {"value": "wynqvrh053", "label": "USERNAME"},
            {"value": "10:20am", "label": "TIME"},
            {"value": "luka.burg", "label": "USERNAME"},
            {"value": "21", "label": "TIME"},
            {"value": "qahil.wittauer", "label": "USERNAME"},
            {"value": "quarter past 13", "label": "TIME"},
            {"value": "gholamhossein.ruschke", "label": "USERNAME"},
            {"value": "9:47 PM", "label": "TIME"},
            {"value": "pdmjrsyoz1460", "label": "USERNAME"},
        ],
        redacted_text="Subject: Group Messaging for Admissions Process\n\nGood morning, everyone,\n\nI hope this message finds you well. As we continue our admissions processes, I would like to update you on the latest developments and key information. Please find below the timeline for our upcoming meetings:\n\n- [USERNAME] - Meeting at [TIME]\n- [USERNAME] - Meeting at [TIME]\n- [USERNAME] - Meeting at [TIME]\n- [USERNAME] - Meeting at [TIME]\n- [USERNAME] ",
    ).with_inputs("text"),
    # 2. DATE, EMAIL, LASTNAME1, LASTNAME2, SOCIALNUMBER, TIME
    dspy.Example(
        text="Subject: Admission Notification - Great Britain University\n\nDear Applicants,\n\nWe are thrilled to inform you about the status of your admission to Great Britain University. Please read the details below for the automated notification.\n\nDate of Notification: 5:24am on August 5th, 2057\n\n**Applicant Details**\n\n1. Applicant: Balloi Eckrich\n   Email: bballoi@yahoo.com\n   Social Number: 996 076 6460\n   ID ",
        entities=[
            {"value": "5:24am", "label": "TIME"},
            {"value": "August 5th, 2057", "label": "DATE"},
            {"value": "Balloi", "label": "LASTNAME1"},
            {"value": "Eckrich", "label": "LASTNAME2"},
            {"value": "bballoi@yahoo.com", "label": "EMAIL"},
            {"value": "996 076 6460", "label": "SOCIALNUMBER"},
        ],
        redacted_text="Subject: Admission Notification - Great Britain University\n\nDear Applicants,\n\nWe are thrilled to inform you about the status of your admission to Great Britain University. Please read the details below for the automated notification.\n\nDate of Notification: [TIME] on [DATE]\n\n**Applicant Details**\n\n1. Applicant: [LASTNAME1] [LASTNAME2]\n   Email: [EMAIL]\n   Social Number: [SOCIALNUMBER]\n   ID ",
    ).with_inputs("text"),
    # 3. BUILDING, CITY, COUNTRY, EMAIL, IDCARD, LASTNAME1, LASTNAME2, PASS, POSTCODE, SOCIALNUMBER, STATE, STREET
    dspy.Example(
        text="Card: KB90324ER\n   Country: GB\n   Building: 163\n   Street: Conygre Grove\n   City: Bristol\n   State: ENG\n   Postcode: BS34 7HU, BS34 7HZ\n   Password: q4R\\\\\n\n2. Applicant: Baasgaran Palmoso\n   Email: blerenbaasgara@gmail.com\n   Social Number: 107-393-9036\n   ID Card: SC78428CU\n   Country: United Kingdom\n   Building: 646\n   Street: School Lane\n   City: Altrincham\n   State: ENG\n   Postcode: WA14 5R",
        entities=[
            {"value": "KB90324ER", "label": "IDCARD"},
            {"value": "GB", "label": "COUNTRY"},
            {"value": "163", "label": "BUILDING"},
            {"value": "Conygre Grove", "label": "STREET"},
            {"value": "Bristol", "label": "CITY"},
            {"value": "ENG", "label": "STATE"},
            {"value": "BS34 7HU, BS34 7HZ", "label": "POSTCODE"},
            {"value": "q4R\\\\", "label": "PASS"},
            {"value": "Baasgaran", "label": "LASTNAME1"},
            {"value": "Palmoso", "label": "LASTNAME2"},
            {"value": "blerenbaasgara@gmail.com", "label": "EMAIL"},
            {"value": "107-393-9036", "label": "SOCIALNUMBER"},
            {"value": "SC78428CU", "label": "IDCARD"},
            {"value": "United Kingdom", "label": "COUNTRY"},
            {"value": "646", "label": "BUILDING"},
            {"value": "School Lane", "label": "STREET"},
            {"value": "Altrincham", "label": "CITY"},
            {"value": "ENG", "label": "STATE"},
        ],
        redacted_text="Card: [IDCARD]\n   Country: [COUNTRY]\n   Building: [BUILDING]\n   Street: [STREET]\n   City: [CITY]\n   State: [STATE]\n   Postcode: [POSTCODE]\n   Password: [PASS]\n\n2. Applicant: [LASTNAME1] [LASTNAME2]\n   Email: [EMAIL]\n   Social Number: [SOCIALNUMBER]\n   ID Card: [IDCARD]\n   Country: [COUNTRY]\n   Building: [BUILDING]\n   Street: [STREET]\n   City: [CITY]\n   State: [STATE]\n   Postcode: WA14 5R",
    ).with_inputs("text"),
    # 4. DATE, PASS
    dspy.Example(
        text='N, WA14 5RW\n   Password: r]iD1#8\n\n...and so on for all applicants listed.\n\nCongratulations to all successful applicants. Please await further communication from the admission office regarding the next steps to finalize your enrollment. We look forward to welcoming you to Great Britain University for the academic year starting on August 5th, 2057.\n\nBest regards,\nAdmissions Team\nGreat Britain University"',
        entities=[
            {"value": "r]iD1#8", "label": "PASS"},
            {"value": "August 5th, 2057", "label": "DATE"},
        ],
        redacted_text='N, WA14 5RW\n   Password: [PASS]\n\n...and so on for all applicants listed.\n\nCongratulations to all successful applicants. Please await further communication from the admission office regarding the next steps to finalize your enrollment. We look forward to welcoming you to Great Britain University for the academic year starting on [DATE].\n\nBest regards,\nAdmissions Team\nGreat Britain University"',
    ).with_inputs("text"),
    # 5. PASSPORT
    dspy.Example(
        text="Subject: Admission Application Attachments Confirmation\n\nDear Applicants,\n\nWe hope this email finds you well. \n\nThis is to confirm that we have received the necessary documentation for your admission applications. Please find attached below the list of attachments for each applicant:\n\nApplicant A:\n- Passport: 301025226\n- Driver's License: ROSAL 955306 9 ",
        entities=[
            {"value": "301025226", "label": "PASSPORT"},
        ],
        redacted_text="Subject: Admission Application Attachments Confirmation\n\nDear Applicants,\n\nWe hope this email finds you well. \n\nThis is to confirm that we have received the necessary documentation for your admission applications. Please find attached below the list of attachments for each applicant:\n\nApplicant A:\n- Passport: [PASSPORT]\n- Driver's License: ROSAL 955306 9 ",
    ).with_inputs("text"),
    # 6. DRIVERLICENSE, EMAIL, PASSPORT, SOCIALNUMBER, TEL
    dspy.Example(
        text="981\n- Social Security Number: 554.575.9355\n- Email: vtpkbqcutaxb799@yahoo.com\n- Telephone: +1322380 4181\n\nApplicant B:\n- Passport: 933932951\n- Driver's License: LOUMA.657200.9.504\n- Social Security Number: 0610780437\n- Email: xwjhgbgg009@outlook.com\n- Telephone: +543 51-082.8035\n\n...and so forth for the remaining applicants.\n\nPlease review the attachment",
        entities=[
            {"value": "554.575.9355", "label": "SOCIALNUMBER"},
            {"value": "vtpkbqcutaxb799@yahoo.com", "label": "EMAIL"},
            {"value": "+1322380 4181", "label": "TEL"},
            {"value": "933932951", "label": "PASSPORT"},
            {"value": "LOUMA.657200.9.504", "label": "DRIVERLICENSE"},
            {"value": "0610780437", "label": "SOCIALNUMBER"},
            {"value": "xwjhgbgg009@outlook.com", "label": "EMAIL"},
            {"value": "+543 51-082.8035", "label": "TEL"},
        ],
        redacted_text="981\n- Social Security Number: [SOCIALNUMBER]\n- Email: [EMAIL]\n- Telephone: [TEL]\n\nApplicant B:\n- Passport: [PASSPORT]\n- Driver's License: [DRIVERLICENSE]\n- Social Security Number: [SOCIALNUMBER]\n- Email: [EMAIL]\n- Telephone: [TEL]\n\n...and so forth for the remaining applicants.\n\nPlease review the attachment",
    ).with_inputs("text"),
    # 7. DATE, STREET, TIME
    dspy.Example(
        text="s carefully and inform us immediately if there are any discrepancies or if you require further assistance. \n\nWe appreciate your cooperation in this process and look forward to reviewing your applications.\n\nWarm regards,\n\nAdmissions Office \n[University Name]  \n[University Address]  \n[City, State, Postcode]   \n[Country, Rue des Écoles]  \n13, September/54  ",
        entities=[
            {"value": "Rue des Écoles", "label": "STREET"},
            {"value": "13", "label": "TIME"},
            {"value": "September/54", "label": "DATE"},
        ],
        redacted_text="s carefully and inform us immediately if there are any discrepancies or if you require further assistance. \n\nWe appreciate your cooperation in this process and look forward to reviewing your applications.\n\nWarm regards,\n\nAdmissions Office \n[University Name]  \n[University Address]  \n[City, State, Postcode]   \n[Country, [STREET]]  \n[TIME], [DATE]  ",
    ).with_inputs("text"),
    # 8. BOD, TEL, USERNAME
    dspy.Example(
        text="- id_1:\n  Feb 8, 1986\n  iloweintögl\n  4929-667-4889\n  Details: Engaged in extracurricular activities throughout high school, showcasing leadership skills. Enthusiastic about pursuing higher education to further develop interpersonal and academic abilities.\n- id_2:\n  26/10/2004\n  abdi\n  076 1352.8018\n  Details: Demonstrated exceptional academic performance in mathematics and sciences, with a keen interest in research and innovation. Activel",
        entities=[
            {"value": "Feb 8, 1986", "label": "BOD"},
            {"value": "iloweintögl", "label": "USERNAME"},
            {"value": "4929-667-4889", "label": "TEL"},
            {"value": "26/10/2004", "label": "BOD"},
            {"value": "abdi", "label": "USERNAME"},
            {"value": "076 1352.8018", "label": "TEL"},
        ],
        redacted_text="- id_1:\n  [BOD]\n  [USERNAME]\n  [TEL]\n  Details: Engaged in extracurricular activities throughout high school, showcasing leadership skills. Enthusiastic about pursuing higher education to further develop interpersonal and academic abilities.\n- id_2:\n  [BOD]\n  [USERNAME]\n  [TEL]\n  Details: Demonstrated exceptional academic performance in mathematics and sciences, with a keen interest in research and innovation. Activel",
    ).with_inputs("text"),
    # 9. STREET, USERNAME
    dspy.Example(
        text='"Dear [1980refad.chaïb], \n\nWe are pleased to inform you that your application has been processed and a decision has been reached regarding your admission status. Please be advised to check your email for detailed information on the outcome of your application.\n\nShould you have any questions or require further assistance, feel free to contact us at [japeschk92] or visit our office at [Preston Road]. \n\nCongratulations on your successful admission!\n\nBest regards,\n[GR]"',
        entities=[
            {"value": "1980refad.chaïb", "label": "USERNAME"},
            {"value": "japeschk92", "label": "USERNAME"},
            {"value": "Preston Road", "label": "STREET"},
            {"value": "GR", "label": "USERNAME"},
        ],
        redacted_text='"Dear [[USERNAME]], \n\nWe are pleased to inform you that your application has been processed and a decision has been reached regarding your admission status. Please be advised to check your email for detailed information on the outcome of your application.\n\nShould you have any questions or require further assistance, feel free to contact us at [[USERNAME]] or visit our office at [[STREET]]. \n\nCongratulations on your successful admission!\n\nBest regards,\n[[USERNAME]]"',
    ).with_inputs("text"),
    # 10. BOD, DATE, DRIVERLICENSE, EMAIL, IDCARD, IP, PASS, POSTCODE, SEX, TIME
    dspy.Example(
        text="Evaluation Report: Candidate Suitability for Admission\n\nDate: 29/06/2013\nTime: 7:59 PM\nLocation: CM21\n\nCandidate A:\n- Sex: M\n- Date of Birth: October/97\n- Email: MVC@tutanota.com\n- ID Card Number: RF69601MW\n- Driver's License: MASCU910077MV815\n- IP Address: 7836:3dcf:9edf:692:fd5f:4de5:a9d6:da24\n- Password: Be~o}.zq8^1\"\n\nCandidate B:\n- Sex: F\n- Date of Birth: 7th August 1963\n- Email: mindkassir@hotmail.com\n- ID Card Number: DE83548AE\n- Driver's ",
        entities=[
            {"value": "29/06/2013", "label": "DATE"},
            {"value": "7:59 PM", "label": "TIME"},
            {"value": "CM21", "label": "POSTCODE"},
            {"value": "M", "label": "SEX"},
            {"value": "October/97", "label": "BOD"},
            {"value": "MVC@tutanota.com", "label": "EMAIL"},
            {"value": "RF69601MW", "label": "IDCARD"},
            {"value": "MASCU910077MV815", "label": "DRIVERLICENSE"},
            {"value": "7836:3dcf:9edf:692:fd5f:4de5:a9d6:da24", "label": "IP"},
            {"value": 'Be~o}.zq8^1"', "label": "PASS"},
            {"value": "F", "label": "SEX"},
            {"value": "7th August 1963", "label": "BOD"},
            {"value": "mindkassir@hotmail.com", "label": "EMAIL"},
            {"value": "DE83548AE", "label": "IDCARD"},
        ],
        redacted_text="Evaluation Report: Candidate Suitability for Admission\n\nDate: [DATE]\nTime: [TIME]\nLocation: [POSTCODE]\n\nCandidate A:\n- Sex: [SEX]\n- Date of Birth: [BOD]\n- Email: [EMAIL]\n- ID Card Number: [IDCARD]\n- Driver's License: [DRIVERLICENSE]\n- IP Address: [IP]\n- Password: [PASS]\n\nCandidate B:\n- Sex: [SEX]\n- Date of Birth: [BOD]\n- Email: [EMAIL]\n- ID Card Number: [IDCARD]\n- Driver's ",
    ).with_inputs("text"),
    # 11. DRIVERLICENSE, EMAIL, IDCARD, IP, PASS
    dspy.Example(
        text='License: MINDA.658073.MR.352\n- IP Address: 1dca:680f:2938:6035:4ed8:81d:c6d6:3b1a\n- Password: "{0w7/U\n\nOther Candidates:\n- Candidate C: Email: asukas55@aol.com, ID Card Number: UK57900JK\n- Candidate D: Email: 3chunmei@protonmail.com, ID Card Number: UGG576437H\n- Candidate E: Email: ydtjqhxrfiv1162@hotmail.com, ID Card Number: OU79828NR\n- Candidate F: Email: A@protonmail.com, ID Card Number: FE15976DV\n- Candidate G: Email: tdjispgtfiqx547@tutanot',
        entities=[
            {"value": "MINDA.658073.MR.352", "label": "DRIVERLICENSE"},
            {"value": "1dca:680f:2938:6035:4ed8:81d:c6d6:3b1a", "label": "IP"},
            {"value": '"{0w7/U', "label": "PASS"},
            {"value": "asukas55@aol.com", "label": "EMAIL"},
            {"value": "UK57900JK", "label": "IDCARD"},
            {"value": "3chunmei@protonmail.com", "label": "EMAIL"},
            {"value": "UGG576437H", "label": "IDCARD"},
            {"value": "ydtjqhxrfiv1162@hotmail.com", "label": "EMAIL"},
            {"value": "OU79828NR", "label": "IDCARD"},
            {"value": "A@protonmail.com", "label": "EMAIL"},
            {"value": "FE15976DV", "label": "IDCARD"},
        ],
        redacted_text="License: [DRIVERLICENSE]\n- IP Address: [IP]\n- Password: [PASS]\n\nOther Candidates:\n- Candidate C: Email: [EMAIL], ID Card Number: [IDCARD]\n- Candidate D: Email: [EMAIL], ID Card Number: [IDCARD]\n- Candidate E: Email: [EMAIL], ID Card Number: [IDCARD]\n- Candidate F: Email: [EMAIL], ID Card Number: [IDCARD]\n- Candidate G: Email: tdjispgtfiqx547@tutanot",
    ).with_inputs("text"),
    # 12. EMAIL, IDCARD
    dspy.Example(
        text="a.com, ID Card Number: RS96293BB\n- Candidate H: Email: N@gmail.com, ID Card Number: BK03584DF\n- Candidate I: Email: 1938qun@hotmail.com, ID Card Number: XW39670SY\n- Candidate J: Email: babitha.iliksoy1969@hotmail.com, ID Card Number: LE33574SP\n\nThis comprehensive evaluation report covers a detailed assessment of each candidate's suitability for admission, taking into account their personal information and background details provided.\n\nThank you.",
        entities=[
            {"value": "RS96293BB", "label": "IDCARD"},
            {"value": "N@gmail.com", "label": "EMAIL"},
            {"value": "BK03584DF", "label": "IDCARD"},
            {"value": "1938qun@hotmail.com", "label": "EMAIL"},
            {"value": "XW39670SY", "label": "IDCARD"},
            {"value": "babitha.iliksoy1969@hotmail.com", "label": "EMAIL"},
            {"value": "LE33574SP", "label": "IDCARD"},
        ],
        redacted_text="a.com, ID Card Number: [IDCARD]\n- Candidate H: Email: [EMAIL], ID Card Number: [IDCARD]\n- Candidate I: Email: [EMAIL], ID Card Number: [IDCARD]\n- Candidate J: Email: [EMAIL], ID Card Number: [IDCARD]\n\nThis comprehensive evaluation report covers a detailed assessment of each candidate's suitability for admission, taking into account their personal information and background details provided.\n\nThank you.",
    ).with_inputs("text"),
    # 13. BUILDING, CITY, COUNTRY, EMAIL, IDCARD, IP, LASTNAME1, POSTCODE, SEX, SOCIALNUMBER, STATE, STREET, TEL, TIME, USERNAME
    dspy.Example(
        text="- Sample Form -\n\n```\n**Student Enrollment Application Form**\n\n**Applicant 1**\n- Full Name: Helbert\n- Gender: F\n- Username: helbert\n- Email: helbert@gmail.com\n- Social Security Number: 080065505\n- ID Card Type: EPY\n- Telephone: 0070-07 986.4979\n- Nationality: Nederland\n- Address: 53, Zantelweg\n- City: Rhoon\n- State: Zuid-Holland\n- Postal Code: 3161\n- IP Address: 7c0c:35a3:e210:d7f1:6366:4206:353a:9ca9\n- Preferred Time: 2:38pm\n\n**Applicant 2**\n- Full Name: Gyo",
        entities=[
            {"value": "Helbert", "label": "LASTNAME1"},
            {"value": "F", "label": "SEX"},
            {"value": "helbert", "label": "USERNAME"},
            {"value": "helbert@gmail.com", "label": "EMAIL"},
            {"value": "080065505", "label": "SOCIALNUMBER"},
            {"value": "EPY", "label": "IDCARD"},
            {"value": "0070-07 986.4979", "label": "TEL"},
            {"value": "Nederland", "label": "COUNTRY"},
            {"value": "53", "label": "BUILDING"},
            {"value": "Zantelweg", "label": "STREET"},
            {"value": "Rhoon", "label": "CITY"},
            {"value": "Zuid-Holland", "label": "STATE"},
            {"value": "3161", "label": "POSTCODE"},
            {"value": "7c0c:35a3:e210:d7f1:6366:4206:353a:9ca9", "label": "IP"},
            {"value": "2:38pm", "label": "TIME"},
        ],
        redacted_text="- Sample Form -\n\n```\n**Student Enrollment Application Form**\n\n**Applicant 1**\n- Full Name: [LASTNAME1]\n- Gender: [SEX]\n- Username: [USERNAME]\n- Email: [EMAIL]\n- Social Security Number: [SOCIALNUMBER]\n- ID Card Type: [IDCARD]\n- Telephone: [TEL]\n- Nationality: [COUNTRY]\n- Address: [BUILDING], [STREET]\n- City: [CITY]\n- State: [STATE]\n- Postal Code: [POSTCODE]\n- IP Address: [IP]\n- Preferred Time: [TIME]\n\n**Applicant 2**\n- Full Name: Gyo",
    ).with_inputs("text"),
    # 14. BUILDING, CITY, COUNTRY, EMAIL, IDCARD, IP, LASTNAME1, LASTNAME2, LASTNAME3, POSTCODE, SECADDRESS, SEX, SOCIALNUMBER, STATE, STREET, TEL, TIME, USERNAME
    dspy.Example(
        text="rgy, Guirard\n- Gender: Masculine\n- Username: kees.gyorgy02\n- Email: keesguirard@aol.com\n- Social Security Number: 464501286\n- ID Card Type: RGI\n- Telephone: 00758-30091\n- Nationality: Nederland\n- Address: 397, Kostverlorenkade\n- City: Amstelveen, State: NH\n- Postal Code: 1183 TM, Secondary Address: PB 73\n- IP Address: 11.47.34.34, Preferred Time: 2:58am\n\n**Applicant 3**\n- Last Name: Potkonjak\n\n**Applicant 4**\n- Last Name: Ucha, Mastrogiacomo, Raizner\n\n**Appl",
        entities=[
            {"value": "Guirard", "label": "LASTNAME2"},
            {"value": "Masculine", "label": "SEX"},
            {"value": "kees.gyorgy02", "label": "USERNAME"},
            {"value": "keesguirard@aol.com", "label": "EMAIL"},
            {"value": "464501286", "label": "SOCIALNUMBER"},
            {"value": "RGI", "label": "IDCARD"},
            {"value": "00758-30091", "label": "TEL"},
            {"value": "Nederland", "label": "COUNTRY"},
            {"value": "397", "label": "BUILDING"},
            {"value": "Kostverlorenkade", "label": "STREET"},
            {"value": "Amstelveen", "label": "CITY"},
            {"value": "NH", "label": "STATE"},
            {"value": "1183 TM", "label": "POSTCODE"},
            {"value": "PB 73", "label": "SECADDRESS"},
            {"value": "11.47.34.34", "label": "IP"},
            {"value": "2:58am", "label": "TIME"},
            {"value": "Potkonjak", "label": "LASTNAME1"},
            {"value": "Ucha", "label": "LASTNAME1"},
            {"value": "Mastrogiacomo", "label": "LASTNAME2"},
            {"value": "Raizner", "label": "LASTNAME3"},
        ],
        redacted_text="rgy, [LASTNAME2]\n- Gender: [SEX]\n- Username: [USERNAME]\n- Email: [EMAIL]\n- Social Security Number: [SOCIALNUMBER]\n- ID Card Type: [IDCARD]\n- Telephone: [TEL]\n- Nationality: [COUNTRY]\n- Address: [BUILDING], [STREET]\n- City: [CITY], State: [STATE]\n- Postal Code: [POSTCODE], Secondary Address: [SECADDRESS]\n- IP Address: [IP], Preferred Time: [TIME]\n\n**Applicant 3**\n- Last Name: [LASTNAME1]\n\n**Applicant 4**\n- Last Name: [LASTNAME1], [LASTNAME2], [LASTNAME3]\n\n**Appl",
    ).with_inputs("text"),
    # 15. CITY, DATE, LASTNAME1, LASTNAME2, LASTNAME3, TIME
    dspy.Example(
        text="icant 5**\n- Last Name: Speil-Ehrenreich\n\n**Applicant 6**\n- Last Name: Ginat\n\n**Applicant 7**\n- Last Name: Kafa, Piccio\n\n**Applicant 8**\n- Last Name: al Najjar, Francioni, Rolny\n\n**Applicant 9**\n- Last Name: El-Gharbawy, Nezirevic\n\n**Applicant 10**\n- Last Name: Ipek, Lelouch-Ferdinand, Jupin\n\n**Applicant 11**\n- Last Name: El Bouchti\n\n**Applicant 12**\n- Last Name: Verzaro\n\n- Background Information -\n- Time: 22:29:58\n- City: Brighton\n- Date: 23rd June 1958\n```",
        entities=[
            {"value": "Speil-Ehrenreich", "label": "LASTNAME1"},
            {"value": "Ginat", "label": "LASTNAME1"},
            {"value": "Kafa", "label": "LASTNAME1"},
            {"value": "Piccio", "label": "LASTNAME2"},
            {"value": "al Najjar", "label": "LASTNAME1"},
            {"value": "Francioni", "label": "LASTNAME2"},
            {"value": "Rolny", "label": "LASTNAME3"},
            {"value": "El-Gharbawy", "label": "LASTNAME1"},
            {"value": "Nezirevic", "label": "LASTNAME2"},
            {"value": "Ipek", "label": "LASTNAME1"},
            {"value": "Lelouch-Ferdinand", "label": "LASTNAME2"},
            {"value": "Jupin", "label": "LASTNAME3"},
            {"value": "El Bouchti", "label": "LASTNAME1"},
            {"value": "Verzaro", "label": "LASTNAME1"},
            {"value": "22:29:58", "label": "TIME"},
            {"value": "Brighton", "label": "CITY"},
            {"value": "23rd June 1958", "label": "DATE"},
        ],
        redacted_text="icant 5**\n- Last Name: [LASTNAME1]\n\n**Applicant 6**\n- Last Name: [LASTNAME1]\n\n**Applicant 7**\n- Last Name: [LASTNAME1], [LASTNAME2]\n\n**Applicant 8**\n- Last Name: [LASTNAME1], [LASTNAME2], [LASTNAME3]\n\n**Applicant 9**\n- Last Name: [LASTNAME1], [LASTNAME2]\n\n**Applicant 10**\n- Last Name: [LASTNAME1], [LASTNAME2], [LASTNAME3]\n\n**Applicant 11**\n- Last Name: [LASTNAME1]\n\n**Applicant 12**\n- Last Name: [LASTNAME1]\n\n- Background Information -\n- Time: [TIME]\n- City: [CITY]\n- Date: [DATE]\n```",
    ).with_inputs("text"),
    # 16. LASTNAME1
    dspy.Example(
        text="Public Comment Thread - Online Course Development\n\n**Thread: Online Learning Strategies**\n🔹 **Comment by 'Andreoni':**  \n\"Hello everyone, I am excited to be part of this course development community. With my background in education, I hope to contribute valuable insights to enhance the learning experience. Looking forward to collaborating with all of you.\"\n\n🔹 **Reply by 'Modrusan':**  \n\"Welcome, 'Andreoni'! Your expertise will undoubtedly enrich our discussions. I am particularly interested i",
        entities=[
            {"value": "Andreoni", "label": "LASTNAME1"},
            {"value": "Modrusan", "label": "LASTNAME1"},
            {"value": "Andreoni", "label": "LASTNAME1"},
        ],
        redacted_text="Public Comment Thread - Online Course Development\n\n**Thread: Online Learning Strategies**\n🔹 **Comment by '[LASTNAME1]':**  \n\"Hello everyone, I am excited to be part of this course development community. With my background in education, I hope to contribute valuable insights to enhance the learning experience. Looking forward to collaborating with all of you.\"\n\n🔹 **Reply by '[LASTNAME1]':**  \n\"Welcome, '[LASTNAME1]'! Your expertise will undoubtedly enrich our discussions. I am particularly interested i",
    ).with_inputs("text"),
    # 17. LASTNAME1, TIME
    dspy.Example(
        text=" 'Pari':**  \n\"Greetings, esteemed colleagues! Bringing a blend of theoretical knowledge and instructional expertise, I am thrilled to participate in shaping the future of online learning. Let's work together to make education accessible and engaging for all.\"\n\n🔹 **Reply by 'Bissinger':**  \n\"Hello, 'Pari'! Your commitment to inclusive education is inspiring. I look forward to exploring strategies that promote diversity and inclusivity in online learning environments.\"\n\n🕒 **Timestamp: 1:21pm**",
        entities=[
            {"value": "Pari", "label": "LASTNAME1"},
            {"value": "Bissinger", "label": "LASTNAME1"},
            {"value": "Pari", "label": "LASTNAME1"},
            {"value": "1:21pm", "label": "TIME"},
        ],
        redacted_text=" '[LASTNAME1]':**  \n\"Greetings, esteemed colleagues! Bringing a blend of theoretical knowledge and instructional expertise, I am thrilled to participate in shaping the future of online learning. Let's work together to make education accessible and engaging for all.\"\n\n🔹 **Reply by '[LASTNAME1]':**  \n\"Hello, '[LASTNAME1]'! Your commitment to inclusive education is inspiring. I look forward to exploring strategies that promote diversity and inclusivity in online learning environments.\"\n\n🕒 **Timestamp: [TIME]**",
    ).with_inputs("text"),
    # 18. BUILDING, CITY, COUNTRY, STATE, STREET
    dspy.Example(
        text="urvey Link: [Survey Link here]\n\nYour input is highly appreciated and will be instrumental in shaping the future of online learning at our institution. The survey should take no more than 15 minutes to complete.\n\nWe value your perspective and look forward to receiving your feedback. Thank you for your time and contribution to our continuous improvement efforts.\n\nWarm regards,\n\n[Course Development Team]  \nCommon Lane, 531, Rochester Upnor, ENG, United Kingdom, 22nd August 2007",
        entities=[
            {"value": "Common Lane", "label": "STREET"},
            {"value": "531", "label": "BUILDING"},
            {"value": "Rochester Upnor", "label": "CITY"},
            {"value": "ENG", "label": "STATE"},
            {"value": "United Kingdom", "label": "COUNTRY"},
        ],
        redacted_text="urvey Link: [Survey Link here]\n\nYour input is highly appreciated and will be instrumental in shaping the future of online learning at our institution. The survey should take no more than 15 minutes to complete.\n\nWe value your perspective and look forward to receiving your feedback. Thank you for your time and contribution to our continuous improvement efforts.\n\nWarm regards,\n\n[Course Development Team]  \n[STREET], [BUILDING], [CITY], [STATE], [COUNTRY], 22nd August 2007",
    ).with_inputs("text"),
    # 19. GIVENNAME1, GIVENNAME2, SEX, TIME
    dspy.Example(
        text='<?xml version="1.0" encoding="UTF-8"?>\n<online_course>\n\t<course>\n\t\t<instructor>\n\t\t\t<name>Ansgar</name>\n\t\t\t<sex>M</sex>\n\t\t\t<time>7 o\'clock</time>\n\t\t</instructor>\n\t\t<students>\n\t\t\t<student>\n\t\t\t\t<name>Délina</name>\n\t\t\t\t<sex>Prefer not to disclose</sex>\n\t\t\t\t<time>3:49am</time>\n\t\t\t</student>\n\t\t\t<student>\n\t\t\t\t<name>Szimonetta</name>\n\t\t\t\t<sex>F</sex>\n\t',
        entities=[
            {"value": "Ansgar", "label": "GIVENNAME1"},
            {"value": "M", "label": "SEX"},
            {"value": "7 o'clock", "label": "TIME"},
            {"value": "Délina", "label": "GIVENNAME2"},
            {"value": "Prefer not to disclose", "label": "SEX"},
            {"value": "3:49am", "label": "TIME"},
            {"value": "Szimonetta", "label": "GIVENNAME1"},
            {"value": "F", "label": "SEX"},
        ],
        redacted_text='<?xml version="1.0" encoding="UTF-8"?>\n<online_course>\n\t<course>\n\t\t<instructor>\n\t\t\t<name>[GIVENNAME1]</name>\n\t\t\t<sex>[SEX]</sex>\n\t\t\t<time>[TIME]</time>\n\t\t</instructor>\n\t\t<students>\n\t\t\t<student>\n\t\t\t\t<name>[GIVENNAME2]</name>\n\t\t\t\t<sex>[SEX]</sex>\n\t\t\t\t<time>[TIME]</time>\n\t\t\t</student>\n\t\t\t<student>\n\t\t\t\t<name>[GIVENNAME1]</name>\n\t\t\t\t<sex>[SEX]</sex>\n\t',
    ).with_inputs("text"),
    # 20. GIVENNAME1, SEX, STREET, TIME
    dspy.Example(
        text="\t\t\t<time>2:05 PM</time>\n\t\t\t</student>\n\t\t\t<student>\n\t\t\t\t<name>Nasnet</name>\n\t\t\t\t<sex>F</sex>\n\t\t\t\t<time>1:28 PM</time>\n\t\t\t</student>\n\t\t\t<student>\n\t\t\t\t<name>Fania</name>\n\t\t\t\t<sex>Female</sex>\n\t\t\t\t<time>04</time>\n\t\t\t</student>\n\t\t</students>\n\t</course>\n\t<location>\n\t\t<street>Chirton Grove</street>\n\t\t<time>11:51am</time>\n\t</location>\n</online_course>",
        entities=[
            {"value": "2:05 PM", "label": "TIME"},
            {"value": "Nasnet", "label": "GIVENNAME1"},
            {"value": "F", "label": "SEX"},
            {"value": "1:28 PM", "label": "TIME"},
            {"value": "Fania", "label": "GIVENNAME1"},
            {"value": "Female", "label": "SEX"},
            {"value": "04", "label": "TIME"},
            {"value": "Chirton Grove", "label": "STREET"},
            {"value": "11:51am", "label": "TIME"},
        ],
        redacted_text="\t\t\t<time>[TIME]</time>\n\t\t\t</student>\n\t\t\t<student>\n\t\t\t\t<name>[GIVENNAME1]</name>\n\t\t\t\t<sex>[SEX]</sex>\n\t\t\t\t<time>[TIME]</time>\n\t\t\t</student>\n\t\t\t<student>\n\t\t\t\t<name>[GIVENNAME1]</name>\n\t\t\t\t<sex>[SEX]</sex>\n\t\t\t\t<time>[TIME]</time>\n\t\t\t</student>\n\t\t</students>\n\t</course>\n\t<location>\n\t\t<street>[STREET]</street>\n\t\t<time>[TIME]</time>\n\t</location>\n</online_course>",
    ).with_inputs("text"),
    # 21. GIVENNAME1
    dspy.Example(
        text='{\n  "course_details": {\n    "course_name": "Online Course Development Masterclass",\n    "instructor": "Iso",\n    "designer": "Liwam",\n    "description": "This advanced course will take you on a journey through the latest strategies and technologies in online course development. Learn from industry experts and enhance your skills to create engaging and effe',
        entities=[
            {"value": "Iso", "label": "GIVENNAME1"},
            {"value": "Liwam", "label": "GIVENNAME1"},
        ],
        redacted_text='{\n  "course_details": {\n    "course_name": "Online Course Development Masterclass",\n    "instructor": "[GIVENNAME1]",\n    "designer": "[GIVENNAME1]",\n    "description": "This advanced course will take you on a journey through the latest strategies and technologies in online course development. Learn from industry experts and enhance your skills to create engaging and effe',
    ).with_inputs("text"),
    # 22. EMAIL, GIVENNAME1, STATE
    dspy.Example(
        text='"4:00 PM"\n        }\n      ],\n      "location": "ENG"\n    },\n    "enrollment_deadline": "2023-08-15",\n    "cost": "$499",\n    "requirements": {\n      "software": "Iso\'s Online Course Creation Tool",\n      "experience": "Basic knowledge of online teaching"\n    },\n    "contacts": {\n      "email": "tiurid@yahoo.com",\n      "phone": "+4402033654632"\n    }\n  }\n}',
        entities=[
            {"value": "ENG", "label": "STATE"},
            {"value": "Iso", "label": "GIVENNAME1"},
            {"value": "tiurid@yahoo.com", "label": "EMAIL"},
        ],
        redacted_text='"4:00 PM"\n        }\n      ],\n      "location": "[STATE]"\n    },\n    "enrollment_deadline": "2023-08-15",\n    "cost": "$499",\n    "requirements": {\n      "software": "[GIVENNAME1]\'s Online Course Creation Tool",\n      "experience": "Basic knowledge of online teaching"\n    },\n    "contacts": {\n      "email": "[EMAIL]",\n      "phone": "+4402033654632"\n    }\n  }\n}',
    ).with_inputs("text"),
    # 23. BOD, DATE, EMAIL, GEOCOORD, IDCARD
    dspy.Example(
        text="since: October 27th, 1998  \n       Email: 27jantima.härlin@aol.com  \n       ID Card: GK82530PG\n\n**Background Information:**\n- Geographic Coordinates: [53.1051, -2.62021]  \n- Date of Update: 27/03/2014\n\nWe hope that this new private messaging feature will facilitate meaningful interactions and encourage collaboration among our users. Feel free to explore this feature and share your feedback with us. Thank you for being part of our community!\n\nBest regards,\nPhoto Sharing Platform Team",
        entities=[
            {"value": "October 27th, 1998", "label": "BOD"},
            {"value": "27jantima.härlin@aol.com", "label": "EMAIL"},
            {"value": "GK82530PG", "label": "IDCARD"},
            {"value": "[53.1051, -2.62021]", "label": "GEOCOORD"},
            {"value": "27/03/2014", "label": "DATE"},
        ],
        redacted_text="since: [BOD]  \n       Email: [EMAIL]  \n       ID Card: [IDCARD]\n\n**Background Information:**\n- Geographic Coordinates: [GEOCOORD]  \n- Date of Update: [DATE]\n\nWe hope that this new private messaging feature will facilitate meaningful interactions and encourage collaboration among our users. Feel free to explore this feature and share your feedback with us. Thank you for being part of our community!\n\nBest regards,\nPhoto Sharing Platform Team",
    ).with_inputs("text"),
    # 24. DRIVERLICENSE, GIVENNAME1, GIVENNAME2, PASS, PASSPORT, TEL, TITLE, USERNAME
    dspy.Example(
        text='<?xml version="1.0" encoding="UTF-8"?>\n<parental_consent_form>\n    <title>King</title>\n    <user_name>ziaei29</user_name>\n    <driver_license>HERIT.010290.HD.630</driver_license>\n    <passport>090177123</passport>\n    <phone_number>+534 045 899.3504</phone_number>\n    <password>Ma=xhjRi!c5`</password>\n    <child_first_name>Heritier</child_first_name>\n    <child_second_name>Dorent</child_second_name>\n    <councillor_title>Cllr</councillor_title>\n    <councillor_user_name>49SL</councillor_user_name>\n ',
        entities=[
            {"value": "King", "label": "TITLE"},
            {"value": "ziaei29", "label": "USERNAME"},
            {"value": "HERIT.010290.HD.630", "label": "DRIVERLICENSE"},
            {"value": "090177123", "label": "PASSPORT"},
            {"value": "+534 045 899.3504", "label": "TEL"},
            {"value": "Ma=xhjRi!c5`", "label": "PASS"},
            {"value": "Heritier", "label": "GIVENNAME1"},
            {"value": "Dorent", "label": "GIVENNAME2"},
            {"value": "Cllr", "label": "TITLE"},
            {"value": "49SL", "label": "USERNAME"},
        ],
        redacted_text='<?xml version="1.0" encoding="UTF-8"?>\n<parental_consent_form>\n    <title>[TITLE]</title>\n    <user_name>[USERNAME]</user_name>\n    <driver_license>[DRIVERLICENSE]</driver_license>\n    <passport>[PASSPORT]</passport>\n    <phone_number>[TEL]</phone_number>\n    <password>[PASS]</password>\n    <child_first_name>[GIVENNAME1]</child_first_name>\n    <child_second_name>[GIVENNAME2]</child_second_name>\n    <councillor_title>[TITLE]</councillor_title>\n    <councillor_user_name>[USERNAME]</councillor_user_name>\n ',
    ).with_inputs("text"),
    # 25. BUILDING, CITY, COUNTRY, DATE, DRIVERLICENSE, EMAIL, GEOCOORD, GIVENNAME1, GIVENNAME2, IP, LASTNAME1, PASSPORT, POSTCODE, SECADDRESS, SOCIALNUMBER, STATE, STREET, TIME, TITLE
    dspy.Example(
        text='"23489, 576-ABC-123, Alderman, xwlkacrakee21@gmail.com, 342.13.0126, V3463556, 731115575, US, 498, Maricopa Highway, Ojai, CA, 93023-9549, Farmhouse 120, cffd:ba6:93e:e61a:76e3:e8c6:f47d:fae5, Mouayed, Simeone, Muccio, 09:47, [37.4353, -86.941], December/02"',
        entities=[
            {"value": "Alderman", "label": "TITLE"},
            {"value": "xwlkacrakee21@gmail.com", "label": "EMAIL"},
            {"value": "342.13.0126", "label": "SOCIALNUMBER"},
            {"value": "V3463556", "label": "DRIVERLICENSE"},
            {"value": "731115575", "label": "PASSPORT"},
            {"value": "US", "label": "COUNTRY"},
            {"value": "498", "label": "BUILDING"},
            {"value": "Maricopa Highway", "label": "STREET"},
            {"value": "Ojai", "label": "CITY"},
            {"value": "CA", "label": "STATE"},
            {"value": "93023-9549", "label": "POSTCODE"},
            {"value": "Farmhouse 120", "label": "SECADDRESS"},
            {"value": "cffd:ba6:93e:e61a:76e3:e8c6:f47d:fae5", "label": "IP"},
            {"value": "Mouayed", "label": "GIVENNAME1"},
            {"value": "Simeone", "label": "GIVENNAME2"},
            {"value": "Muccio", "label": "LASTNAME1"},
            {"value": "09:47", "label": "TIME"},
            {"value": "[37.4353, -86.941]", "label": "GEOCOORD"},
            {"value": "December/02", "label": "DATE"},
        ],
        redacted_text='"23489, 576-ABC-123, [TITLE], [EMAIL], [SOCIALNUMBER], [DRIVERLICENSE], [PASSPORT], [COUNTRY], [BUILDING], [STREET], [CITY], [STATE], [POSTCODE], [SECADDRESS], [IP], [GIVENNAME1], [GIVENNAME2], [LASTNAME1], [TIME], [GEOCOORD], [DATE]"',
    ).with_inputs("text"),
]
