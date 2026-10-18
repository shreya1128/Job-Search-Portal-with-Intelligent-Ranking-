"""A tiny, hand-made job collection (NOT real data) for the Phase 6 unit tests."""
import numpy as np

from jobintel.ir.index import IndexDoc, InvertedIndex

JOBS = [
    ("Data Analyst", "Pune", ["sql", "excel", "power bi", "data analysis"], 1, 3, "IT"),
    ("Senior Data Analyst", "Pune", ["sql", "excel", "tableau", "data analysis", "python"], 4, 8, "IT"),
    ("Data Scientist", "Bengaluru", ["python", "machine learning", "sql", "statistics"], 2, 6, "IT"),
    ("Machine Learning Engineer", "Pune", ["python", "machine learning", "tensorflow", "docker"], 3, 7, "IT"),
    ("Backend Developer", "Mumbai", ["java", "spring boot", "sql", "docker"], 2, 5, "IT"),
    ("Java Developer", "Pune", ["java", "spring boot", "microservices", "sql"], 3, 6, "IT"),
    ("Frontend Developer", "Mumbai", ["javascript", "react", "css", "html"], 1, 4, "IT"),
    ("Accountant", "Pune", ["tally", "excel", "gst", "accounting"], 1, 4, "Finance"),
    ("Tax Accountant", "Mumbai", ["gst", "accounting", "excel", "taxation"], 2, 5, "Finance"),
    ("Financial Analyst", "Pune", ["excel", "forecasting", "financial analysis", "sql"], 2, 5, "Finance"),
    ("DevOps Engineer", "Bengaluru", ["docker", "kubernetes", "aws", "jenkins"], 3, 8, "IT"),
    ("Warehouse Assistant", "Pune", [], float("nan"), float("nan"), "Logistics"),
]


def make_index(jobs=JOBS):
    docs = []
    for i, (title, city, skills, lo, hi, cat) in enumerate(jobs):
        meta = {"source": "test", "title": title, "company": f"Co{i % 4}", "city": city, "cities": [city], "country": "India", "exp_min": lo, "exp_max": hi, "sal_min": None,
                "sal_max": None, "sal_cur": None, "sal_period": None, "employment": "full-time", "cluster": "other", "category": cat, "text_quality": "full", "skills": skills,
                "posting_date": None, "seniority": None, "external_id": str(i)}
        docs.append(IndexDoc(100 + i, title.lower().split() + skills, meta))
    return InvertedIndex.build(docs)


def make_family_index(n=400, seed=0, families=8, per_family=6):
    """n synthetic jobs: each draws 4-6 skills from one family of `per_family` skills (+ sometimes one from another family)."""
    rng = np.random.default_rng(seed)
    fam = [[f"f{f}_s{i}" for i in range(per_family)] for f in range(families)]
    jobs = []
    for j in range(n):
        f = int(rng.integers(families))
        sk = list(rng.choice(fam[f], size=int(rng.integers(4, 7)), replace=False))
        if rng.random() < 0.3:
            sk.append(str(rng.choice(fam[int(rng.integers(families))])))
        jobs.append((f"role {f} {j}", ["Pune", "Mumbai", "Bengaluru"][j % 3], sorted(set(sk)), 1, 5, f"fam{f}"))
    return make_index(jobs)


def make_text_engine():
    """a SearchEngine over the hand-made JOBS (text = title + skills) so that profile-derived IR queries have something to retrieve."""
    from jobintel.extraction.extractor import Extractor
    from jobintel.extraction.lexicon import SkillLexicon
    from jobintel.ir.engine import MemoryDocStore, SearchEngine
    from jobintel.processing.text import index_tokens
    lex = SkillLexicon.from_seed()
    docs, store = [], {}
    for i, (title, city, skills, lo, hi, cat) in enumerate(JOBS):
        text = f"{title}\n" + ", ".join(skills) + f"\nWe are hiring a {title} in {city}."
        meta = {"source": "test", "title": title, "company": f"Co{i % 4}", "city": city, "cities": [city], "country": "India", "exp_min": lo, "exp_max": hi, "sal_min": None, "sal_max": None,
                "sal_cur": None, "sal_period": None, "employment": "full-time", "cluster": "other", "category": cat, "text_quality": "full", "skills": skills, "posting_date": None,
                "seniority": None, "external_id": str(i)}
        docs.append(IndexDoc(100 + i, index_tokens(text), meta))
        store[100 + i] = (title, text)
    return SearchEngine(InvertedIndex.build(docs), MemoryDocStore(store), Extractor(lex, cities=["Pune", "Mumbai", "Bengaluru"]), lexicon=lex)
