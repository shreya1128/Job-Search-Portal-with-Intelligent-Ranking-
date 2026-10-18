"""A small, hand-made site (NOT real data) for the web tests: a search engine over 36 made-up postings and a SQLite database holding the same postings, so that pages that read both
(search, job details) can be tested without MySQL, the real index or any trained model."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from jobintel.db import models as M
from jobintel.extraction.extractor import Extractor
from jobintel.extraction.lexicon import SkillLexicon
from jobintel.ir.engine import MemoryDocStore, SearchEngine
from jobintel.ir.index import IndexDoc, InvertedIndex
from jobintel.processing.text import index_tokens
from jobintel.rec.catalog import ItemCatalog
from jobintel.web import create_app
from jobintel.web.backend import Backend
from jobintel.web.settings import WebSettings

BASE_ID = 100
CITIES = ["Pune", "Mumbai", "Bengaluru"]
# (title, city, skills, exp_min, exp_max, category, cluster)
BASE = [
    ("Data Analyst", "Pune", ["sql", "excel", "power bi", "data analysis"], 1, 3, "IT", "data_ai"),
    ("Senior Data Analyst", "Pune", ["sql", "excel", "tableau", "data analysis", "python"], 4, 8, "IT", "data_ai"),
    ("Data Scientist", "Bengaluru", ["python", "machine learning", "sql", "statistics"], 2, 6, "IT", "data_ai"),
    ("Machine Learning Engineer", "Pune", ["python", "machine learning", "tensorflow", "docker"], 3, 7, "IT", "data_ai"),
    ("Backend Developer", "Mumbai", ["java", "spring boot", "sql", "docker"], 2, 5, "IT", "software_it"),
    ("Accountant", "Pune", ["tally", "excel", "gst", "accounting"], 1, 4, "Finance", "finance"),
    ("Warehouse Assistant", "Pune", [], float("nan"), float("nan"), "Logistics", "logistics_transport"),
]
EXTRA = [(f"Reporting Analyst {n}", CITIES[n % 3], ["sql", "excel", "reporting"], float(n % 6), float(n % 6 + 2), "IT", "data_ai") for n in range(29)]
JOBS = BASE + EXTRA
FRAUD_ID = BASE_ID + len(BASE) - 1                  # the warehouse job is flagged as fake in the database
APPLY_ID = BASE_ID                                  # has an https link
EVIL_ID = BASE_ID + 1                               # has a javascript: link that must never be rendered
NO_COMPANY_ID = BASE_ID + 2                         # company name is a generated placeholder


def build_engine() -> SearchEngine:
    lex = SkillLexicon.from_seed()
    docs, store = [], {}
    for i, (title, city, skills, lo, hi, cat, cluster) in enumerate(JOBS):
        text = f"{title}\n" + ", ".join(skills) + f"\nWe are hiring a {title} in {city}. You will work with {', '.join(skills) or 'a small team'}."
        meta = {"source": "emscad" if i % 2 else "linkedin", "title": title, "company": f"Co{i % 4}", "city": city, "cities": [city], "country": "India", "exp_min": lo, "exp_max": hi,
                "sal_min": None, "sal_max": None, "sal_cur": None, "sal_period": None, "employment": "full-time", "cluster": cluster, "category": cat, "text_quality": "full",
                "skills": skills, "posting_date": None, "seniority": None, "external_id": str(i)}
        docs.append(IndexDoc(BASE_ID + i, index_tokens(text), meta))
        store[BASE_ID + i] = (title, text)
    return SearchEngine(InvertedIndex.build(docs), MemoryDocStore(store), Extractor(lex, cities=CITIES), lexicon=lex)


def fill_db(engine) -> None:
    with Session(engine) as s:
        src = {n: M.CrawlerSource(name=n, source_type="dataset") for n in ("emscad", "linkedin")}
        s.add_all(src.values())
        s.flush()
        companies = [M.Company(name=f"Co{k}", normalized_name=f"co{k}") for k in range(3)] + [M.Company(name="Company-4c8196", normalized_name="company-4c8196", name_is_derived=True)]
        s.add_all(companies)
        s.flush()
        skills = {}
        for title, city, sk, *_ in JOBS:
            for name in sk:
                if name not in skills:
                    skills[name] = M.Skill(name=name, canonical_name=name, skill_type="technical", source="test")
        s.add_all(skills.values())
        s.flush()
        for i, (title, city, sk, lo, hi, cat, cluster) in enumerate(JOBS):
            jid = BASE_ID + i
            url = "https://example.com/apply/1" if jid == APPLY_ID else "javascript:alert(1)" if jid == EVIL_ID else None
            company = companies[3] if jid == NO_COMPANY_ID else companies[i % 3]
            s.add(M.Job(job_id=jid, source_id=src["emscad" if i % 2 else "linkedin"].source_id, external_id=str(i), title=title, company_id=company.company_id, city=city, cities=[city],
                        country="india", description_clean=f"{title}\n\nWe are hiring a {title} in {city}. You will work with {', '.join(sk) or 'a small team'}.", n_tokens=40, text_quality="full",
                        experience_min=None if lo != lo else lo, experience_max=None if hi != hi else hi, employment_type="full-time", cluster=cluster, url=url,
                        is_fraudulent=(jid == FRAUD_ID)))
        s.flush()
        for i, (title, city, sk, *_) in enumerate(JOBS):
            for name in sk:
                s.add(M.JobSkill(job_id=BASE_ID + i, skill_id=skills[name].skill_id, origin="dataset_tag", mentions=1))
        s.commit()


def make_app(engine, tmp_path: Path, **settings):
    eng = build_engine()
    fill_db(engine)
    st = WebSettings(testing=True, secret_key="test-secret", warmup=False, data_dir=tmp_path, **settings)
    be = Backend(st, db=engine, index=eng.index, engine=eng, catalog=ItemCatalog(eng.index))
    app = create_app(st, be)
    return app, be


def csrf_of(client) -> str:
    """The CSRF token of this client's session (set by any page that renders a form or the layout)."""
    client.get("/about")                      # a public page that always renders the layout (a logged-in user is redirected away from /login)
    with client.session_transaction() as s:
        return s["_csrf"]


def register(client, username="tester", email="tester@example.com", password="correct horse 42"):
    tok = csrf_of(client)
    return client.post("/register", data={"csrf_token": tok, "username": username, "email": email, "password": password, "confirm": password})
