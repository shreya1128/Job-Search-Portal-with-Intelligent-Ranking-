"""SYNTHETIC offline sample jobs (template generated, seed-controlled) - the app works with no internet and no big datasets.

HONESTY NOTE: these postings are generated FROM the curated knowledge graph (jobintel/ingest/curated_kb.py), so
any retrieval/recommendation experiment on them would be circular.  They are labelled `synthetic` everywhere and must
never be used for the evaluation chapters - use the real datasets for that."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from jobintel.ingest.curated_kb import ROLES

CITIES = ["Pune", "Bengaluru", "Mumbai", "Hyderabad", "Chennai", "Delhi NCR", "Gurugram", "Noida", "Kolkata", "Ahmedabad"]
COMPANIES = ["Acme Analytics", "BlueRiver Technologies", "Cobalt Systems", "DeltaWorks Software", "EverGreen Health", "FinSight Capital",
             "GlobalSoft Solutions", "Helix Biosciences", "Indigo Cloud", "Juniper Data Labs", "Kestrel Networks", "Lumen Retail Tech",
             "Meridian Finance", "NovaCare Hospitals", "Orbit Digital", "PixelForge Studios", "Quanta Insights", "Redwood Banking Group",
             "Summit Medical Centre", "TerraByte Infra", "Unity Care Clinics", "Vertex AI Works", "WaveLength Telecom", "Xenon Robotics",
             "YellowBrick Consulting", "Zenith Payments"]
ROLE_CLUSTER = {"Machine Learning Engineer": "data_ai", "Data Scientist": "data_ai", "Data Analyst": "data_ai", "Data Engineer": "data_ai",
                "AI Engineer": "data_ai", "NLP Engineer": "data_ai", "Computer Vision Engineer": "data_ai", "Business Intelligence Developer": "data_ai",
                "Software Engineer": "software_it", "Backend Developer": "software_it", "Frontend Developer": "software_it",
                "Full Stack Developer": "software_it", "DevOps Engineer": "software_it", "Cloud Engineer": "software_it",
                "Database Administrator": "software_it", "QA Engineer": "software_it", "Cybersecurity Analyst": "software_it",
                "Financial Analyst": "finance", "Accountant": "finance", "Registered Nurse": "healthcare"}
FIELD = {"data_ai": ["Computer Science", "Statistics", "Mathematics", "Data Science"], "software_it": ["Computer Science", "Information Technology", "Software Engineering"],
         "finance": ["Finance", "Commerce", "Accounting"], "healthcare": ["Nursing"]}
EXP_BANDS = [(0, 2), (1, 3), (2, 5), (3, 6), (5, 8), (7, 12)]
SENIORITY = {0: "Entry level", 1: "Associate", 2: "Mid-Senior level", 3: "Mid-Senior level", 4: "Senior", 5: "Director"}
RESP = {
    "data_ai": ["Build, train and evaluate models that solve real business problems using {a}, {b} and {c}.", "Clean and analyse large datasets to uncover trends and insights.",
                "Present findings to stakeholders through clear dashboards and reports.", "Deploy and monitor models in production with {c}.", "Work closely with product and engineering teams."],
    "software_it": ["Design, develop and maintain scalable applications using {a}, {b} and {c}.", "Write clean, tested and well documented code.", "Participate in code reviews and agile ceremonies.",
                    "Troubleshoot and resolve production issues.", "Collaborate with designers, QA and product owners."],
    "finance": ["Prepare accurate financial statements, forecasts and variance analyses using {a} and {b}.", "Support month-end close and statutory reporting.", "Review transactions for compliance with {c}.",
                "Partner with business teams on budgeting.", "Identify process improvements."],
    "healthcare": ["Provide safe, compassionate patient care and monitor patient condition.", "Administer medication and treatments as prescribed, following {a}.", "Document care accurately in {b}.",
                   "Coordinate with physicians and the care team.", "Maintain current {c} certification."]}
BENEFITS = ["Health insurance for you and your family", "Flexible working hours and hybrid work", "Learning budget for courses and certifications", "Performance bonus and paid leave"]


def generate(n: int = 300, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    roles = list(ROLES)
    jobs = []
    for i in range(n):
        role = roles[i % len(roles)]
        cl = ROLE_CLUSTER[role]
        ess, opt = ROLES[role]
        skills = ess + rng.sample(opt, k=min(len(opt), rng.randint(2, 4)))
        lo, hi = rng.choice(EXP_BANDS)
        sen = rng.choice(["Senior ", "Junior ", "Lead ", ""]) if cl != "healthcare" else ""
        title = f"{sen}{role}".strip()
        city, company = rng.choice(CITIES), rng.choice(COMPANIES)
        a, b, c = (skills + skills)[:3]
        resp = [t.format(a=a, b=b, c=c) for t in RESP[cl]]
        exp_phrase = rng.choice([f"{lo}-{hi} years of experience", f"minimum {max(lo,1)} years' experience", f"{max(hi,1)}+ years experience", f"at least {max(lo,1)} years of experience"])
        field = rng.choice(FIELD[cl])
        degree = rng.choice([f"Bachelor's degree in {field}", f"B.Tech or MCA in {field}" if cl != "healthcare" else "BSc Nursing", f"Master's degree in {field} is a plus"])
        sal_lo = 3 + lo * 2 + rng.randint(0, 3)
        salary_line = rng.choice([f"Compensation: Rs. {sal_lo}-{sal_lo + rng.randint(3, 8)} LPA", "", ""])
        text = (f"About the role:\n{company} is looking for a {title} to join our {cl.replace('_', ' / ')} team in {city}. "
                f"You will work on meaningful problems with a collaborative team.\n\nResponsibilities:\n" + "\n".join("• " + r for r in resp) +
                f"\n\nRequirements:\n• {exp_phrase.capitalize()} as a {role}\n• {degree}\n• Strong knowledge of {', '.join(skills[:-1])} and {skills[-1]}\n• Good communication and teamwork skills\n\n"
                f"Benefits:\n" + "\n".join("• " + x for x in rng.sample(BENEFITS, 3)) + (f"\n{salary_line}" if salary_line else "") + "\nFull-time, permanent position.")
        jobs.append(dict(
            source="sample_synthetic", external_id=f"syn-{i:04d}", title=title, company=company, location_raw=city, city=city, cities=[city], country="India",
            description=text, skills=skills, experience_min=float(lo), experience_max=float(hi), experience_text=f"{lo}-{hi} years", experience_source="dataset",
            education=degree.split(" is a plus")[0], employment_type="Full-time", seniority=SENIORITY[EXP_BANDS.index((lo, hi))], category=cl,
            posting_date=f"2025-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}", text_quality="full"))
    return jobs


def write_jsonl(path: Path, jobs: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for j in jobs:
            fh.write(json.dumps(j, ensure_ascii=False) + "\n")
