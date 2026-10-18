#!/usr/bin/env python3
"""Download the datasets that are hosted on GitHub into data/raw/ (no Kaggle account needed).

    python scripts/fetch_datasets.py                 # mcf + esco
    python scripts/fetch_datasets.py --only esco
    python scripts/fetch_datasets.py --only emscad   # fallback if you do not have fake_job_postings.csv from Kaggle

NOT downloadable automatically (needs a login / a manual download - see docs/DATASETS.md):
    Naukri CSV (Kaggle), O*NET Excel database (onetcenter.org), Resume.csv (Kaggle / Hugging Face), LinkedIn 1.3M (Kaggle).
Third-party data: check each source's licence/terms and cite it in your report (docs/DATASETS.md lists them)."""
import argparse
import io
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import RAW_DIR

REPOS = {
    "mcf": ("WING-NUS/JD2Skills-BERT-XMLC", "MyCareersFuture job postings (Singapore, 2019) - from the JD2Skills repository"),
    "esco": ("Y4SSERk/ESCO-csv2Graph", "ESCO v1.x English CSV files (skills, occupations, relations) - European Commission, free to use with attribution"),
    "emscad": ("raymondmusyoka/NLP--Fake-Job-Description-Prediction", "EMSCAD / Fake Job Postings CSV (17,880 postings)"),
}


# The original MCF repository (WING-NUS/JD2Skills-BERT-XMLC) was removed (HTTP 404) after the project started; a fork with the same layout is used as a FALLBACK.  Its contents were
# checked against the project's own Phase 1 report: 20,298 records / 18,733 unique / dates 2018-07-09..2019-06-18 / 86% with salary - identical to the original numbers
# (a statistical match, not a byte-for-byte comparison).
FALLBACKS = {"mcf": ["AKBakshay/JD2Skills-BERT-XMLC"]}
EXPECTED_RECORDS = {"mcf": 20298}


def download_zip_any(name: str):
    """try the primary repository, then the fallbacks; returns (zipfile, repository used)."""
    errors = []
    for repo in [REPOS[name][0]] + FALLBACKS.get(name, []):
        try:
            return download_zip(repo), repo
        except Exception as exc:                                  # HTTP 403 / 404, timeouts, bad zip ...
            errors.append(f"{repo}: {exc}")
            print(f"  could not use {repo}: {exc}")
    raise RuntimeError("no repository could be downloaded - " + "; ".join(errors))


def download_zip(repo: str) -> zipfile.ZipFile:
    url = f"https://github.com/{repo}/archive/HEAD.zip"
    print(f"  downloading {url} ...")
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(r.content))


def member(z: zipfile.ZipFile, suffix: str) -> str:
    hits = [n for n in z.namelist() if n.endswith(suffix)]
    if not hits:
        raise FileNotFoundError(f"{suffix} not found in the downloaded repository")
    return hits[0]


def fetch(name: str) -> None:
    repo, desc = REPOS[name]
    print(f"[{name}] {desc}")
    z, used = download_zip_any(name)
    if used != repo:
        print(f"  NOTE: using the fallback repository {used} (the primary one is unavailable)")
    if name == "mcf":
        out = RAW_DIR / "mycareersfuture"
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(z.read(member(z, "data/mycareersfuture.tar.gz"))), mode="r:gz") as tf:
            m = next(x for x in tf.getmembers() if x.name.endswith(".json"))
            (out / "mycareersfuture.json").write_bytes(tf.extractfile(m).read())
        print("  ->", out / "mycareersfuture.json")
        import json
        n = len(json.loads((out / "mycareersfuture.json").read_bytes())["jobs"])
        print(f"  {n:,} postings" + ("" if n == EXPECTED_RECORDS["mcf"] else f"  WARNING: the project was built on {EXPECTED_RECORDS['mcf']:,} - this is a different version of the dataset"))
    elif name == "esco":
        out = RAW_DIR / "esco"
        out.mkdir(parents=True, exist_ok=True)
        for f in ("skills_en.csv", "occupations_en.csv", "occupationSkillRelations_en.csv", "skillSkillRelations_en.csv", "digitalSkillsCollection_en.csv", "skillsHierarchy_en.csv"):
            try:
                (out / f).write_bytes(z.read(member(z, "/" + f)))
            except FileNotFoundError:
                print("  (not in this repository version:", f + ")")
        print("  ->", out)
    elif name == "emscad":
        out = RAW_DIR / "emscad"
        out.mkdir(parents=True, exist_ok=True)
        inner = zipfile.ZipFile(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith(".zip")))))
        (out / "fake_job_postings.csv").write_bytes(inner.read(member(inner, "fake_job_postings.csv")))
        print("  ->", out / "fake_job_postings.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(REPOS), default=["mcf", "esco"])
    failed = []
    for n in ap.parse_args().only:
        try:
            fetch(n)
        except Exception as exc:       # keep going: one failing source must not block the others ...
            print(f"  FAILED: {exc}")
            failed.append(n)
    if failed:                         # ... but the run must not LOOK successful (a silent failure left a whole session without MCF)
        print(f"\n*** NOT DOWNLOADED: {', '.join(failed)} - the data set(s) will be missing downstream; fix the network / repository and re-run ***")
        sys.exit(1)
