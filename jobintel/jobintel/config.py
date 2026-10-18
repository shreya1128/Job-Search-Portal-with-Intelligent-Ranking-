"""Central configuration. Every path and credential comes from environment variables / .env.
No passwords or secrets are stored in the code base."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
if load_dotenv:
    load_dotenv(ROOT / ".env")


def _path(env: str, default: Path) -> Path:
    return Path(os.getenv(env) or default)


DATA_DIR = _path("DATA_DIR", ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
FIXTURE_DIR = DATA_DIR / "fixture_site"
MODELS_DIR = _path("MODELS_DIR", ROOT / "models")

DATASET_PATHS = {
    "naukri": _path("NAUKRI_CSV", RAW_DIR / "naukri" / "naukri_data_science_jobs_india.csv"),
    "emscad": _path("EMSCAD_CSV", RAW_DIR / "emscad" / "fake_job_postings.csv"),
    "mcf": _path("MCF_JSON", RAW_DIR / "mycareersfuture" / "mycareersfuture.json"),
    "linkedin": _path("LINKEDIN_CSV", RAW_DIR / "linkedin" / "linkedin_sample.csv"),
    "linkedin_meta": _path("LINKEDIN_META_CSV", RAW_DIR / "linkedin" / "linkedin_sample_meta.csv"),
    "esco_dir": _path("ESCO_DIR", RAW_DIR / "esco"),
    "onet_dir": _path("ONET_DIR", RAW_DIR / "onet"),
    "resumes": _path("RESUMES_CSV", RAW_DIR / "resumes" / "Resume.csv"),
    "sample": SAMPLE_DIR / "sample_jobs.jsonl",
}

LEXICON_PATH = PROCESSED_DIR / "skill_lexicon.json"
GAZETTEER_PATH = PROCESSED_DIR / "location_gazetteer.json"


def database_url() -> str:
    """MySQL by default (the project requirement). SQLite is allowed only for quick local tests."""
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    if os.getenv("DB_BACKEND", "mysql").lower() == "sqlite":
        return f"sqlite:///{DATA_DIR / 'jobintel.sqlite'}"
    user = os.getenv("DB_USER", "jobintel")
    pwd = quote_plus(os.getenv("DB_PASSWORD", ""))
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME", "jobintel")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{name}?charset=utf8mb4"
