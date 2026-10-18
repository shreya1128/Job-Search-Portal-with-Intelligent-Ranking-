import datetime as dt

import pytest

from jobintel.extraction.extractor import Extractor
from jobintel.extraction.lexicon import SkillLexicon
from jobintel.rec.resume import ResumeAnalyzer, merge_months

TODAY = dt.date(2024, 1, 15)
RESUME = """Priya Sharma
priya.sharma@example.com | +91 98765 43210 | linkedin.com/in/priyasharma | Pune, India

SUMMARY
Data analyst with 3 years of experience in SQL, Excel and Power BI.

EXPERIENCE
Data Analyst, Acme Analytics Pvt Ltd   Jan 2021 - Present
- Built dashboards in Power BI using SQL and Python for the sales team
Junior Analyst, Beta Technologies   06/2019 - 12/2020
- Cleaned data in Excel and wrote SQL reports

EDUCATION
B.Tech in Computer Science, Savitribai Phule Pune University   2015 - 2019

SKILLS
Python, SQL, Excel, Power BI, Tableau
"""


@pytest.fixture(scope="module")
def an():
    return ResumeAnalyzer(Extractor(SkillLexicon.from_seed(), cities=["pune", "mumbai"]))


def test_sections_found(an):
    p = an.analyze(RESUME, today=TODAY)
    assert set(p.sections) >= {"summary", "experience", "education", "skills"}
    assert "Power BI" in p.sections["skills"]


def test_contact_entities_and_name(an):
    p = an.analyze(RESUME, today=TODAY)
    assert p.emails == ["priya.sharma@example.com"]
    assert any("98765" in x for x in p.phones)
    assert any("linkedin.com/in/priyasharma" in u for u in p.urls)
    assert p.name == "Priya Sharma"
    labels = {e.label for e in p.entities}
    assert {"EMAIL", "PHONE", "URL", "PERSON", "ORG", "DATE_RANGE"} <= labels
    assert all(RESUME[e.start:e.end].strip() == e.text.strip() for e in p.entities)          # spans point at the real text
    assert "Acme Analytics Pvt Ltd" in p.organisations and "Beta Technologies" in p.organisations


def test_experience_union_of_ranges_excludes_education(an):
    p = an.analyze(RESUME, today=TODAY)
    # Jan 2021 - Jan 2024 (37 months) + 06/2019 - 12/2020 (19 months) = 56 months; the 2015 - 2019 degree is NOT experience
    assert p.computed_experience_years == pytest.approx(56 / 12, abs=0.05)
    edu = [r for r in p.date_ranges if "2015" in r["text"]][0]
    assert edu["counted"] is False and "education" in edu["reason"]


def test_stated_experience_wins_and_both_are_reported(an):
    p = an.analyze(RESUME, today=TODAY)
    assert p.stated_experience_years == 3.0 and p.experience_years == 3.0 and p.experience_source == "stated"
    assert p.computed_experience_years is not None


def test_skills_weights_distinguish_listed_from_mentioned(an):
    p = an.analyze(RESUME, today=TODAY)
    w = p.skill_weights()
    assert {"python", "sql"} <= {k.lower() for k in w}
    assert max(w.values()) == 1.0
    ev = {s["name"].lower(): s["evidence"] for s in p.skills}
    assert "skills section" in ev["python"]


def test_overlapping_ranges_are_merged_not_double_counted():
    assert merge_months([(0, 24), (12, 36)]) == 36
    assert merge_months([(0, 12), (24, 36)]) == 24
    assert merge_months([]) == 0


def test_year_only_ranges_and_present(an):
    text = "Ravi Kumar\nEXPERIENCE\nEngineer at Foo Systems 2018 - 2021\nSenior Engineer at Bar Labs 2021 - Present\nSKILLS\nPython"
    p = an.analyze(text, today=dt.date(2024, 1, 1))
    assert p.computed_experience_years == pytest.approx(6.0, abs=0.1)        # 2018->2021 = 3 y, 2021->Jan 2024 = 3 y


def test_no_dates_means_unknown_not_zero(an):
    p = an.analyze("Some Person\nSKILLS\nPython and SQL are what I know. " + "word " * 50, today=TODAY)
    assert p.experience_years is None and any("unknown" in w for w in p.warnings)


def test_short_text_warns_and_garbage_does_not_crash(an):
    p = an.analyze("hello", today=TODAY)
    assert any("very short" in w for w in p.warnings) and p.skills == []
    an.analyze("", today=TODAY)
    an.analyze("\x00\x01 ----- @@@@ 2020 - 2019 - 3000", today=TODAY)


def test_implausible_future_or_reversed_ranges_are_not_counted(an):
    p = an.analyze("Test Person\nEXPERIENCE\nX Corp Jan 2030 - Jan 2031\nY Systems 2020 - 2019\n" + "word " * 50, today=TODAY)
    assert p.computed_experience_years is None and all(not r["counted"] for r in p.date_ranges)


def test_unsupported_file_types_are_refused(an, tmp_path):
    f = tmp_path / "cv.pdf"
    f.write_bytes(b"%PDF")
    with pytest.raises(ValueError):
        an.analyze_file(f)
    t = tmp_path / "cv.txt"
    t.write_text(RESUME)
    assert an.analyze_file(t, today=TODAY).name == "Priya Sharma"


def test_certifications_are_single_lines_and_roles_are_unique(an):
    text = RESUME + "\nCERTIFICATIONS\n- Power BI Data Analyst Associate\n- AWS Cloud Practitioner\n"
    p = an.analyze(text, today=TODAY)
    assert all("\n" not in c for c in p.certifications)
    assert "Power BI Data Analyst Associate" in p.certifications and "AWS Cloud Practitioner" in p.certifications
    assert len({r.lower() for r in p.roles}) == len(p.roles)


def test_organisation_rules_institute_of_degrees_certificates_and_line_breaks(an):
    text = ("Ravi Kumar\nEXPERIENCE\nEngineer, Foo Systems   2019 - 2022\nEDUCATION\nB.E. in Information Technology, Pune Institute of Computer Technology   2015 - 2019\n"
            "SKILLS\nPython\nCERTIFICATIONS\n- Google Data Analytics Certificate\n" + "word " * 40)
    orgs = an.analyze(text, today=TODAY).organisations
    assert "Foo Systems" in orgs and "Pune Institute of Computer Technology" in orgs                    # 'X Institute of Y' is one organisation
    assert not ({"Information Technology", "Computer Technology", "Pune Institute", "Google Data Analytics"} & set(orgs))   # degrees / certificates / fragments are not organisations
    assert all("\n" not in o for o in orgs)
