"""Phase 7 verification script.

Checks the environment, database, data files, and all public web routes.
Prints one line per check: OK / FAIL.  Exits 0 only if every check passes.

Run from the project root:
    python scripts/verify_phase7.py

Set JOBINTEL_DATA_DIR to override the default data directory (data/).
Set JOBINTEL_DB_URL to override the default SQLite URL (sqlite:///jobintel.db).
"""
from __future__ import annotations

import importlib
import io
import os
import sys
import traceback
from pathlib import Path

# Force UTF-8 output on Windows (avoids UnicodeEncodeError for arrow/emoji characters)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ------------------------------------------------------------------------------- paths
SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = Path(os.environ.get("JOBINTEL_DATA_DIR", ROOT / "data"))
DB_URL = os.environ.get("JOBINTEL_DB_URL", f"sqlite:///{ROOT / 'jobintel.db'}")

PASS = "\033[32mOK\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

results: list[tuple[str, bool, str]] = []   # (label, ok, detail)


def check(label: str, ok: bool, detail: str = "") -> None:
    tag = PASS if ok else FAIL
    print(f"  [{tag}] {label}" + (f": {detail}" if detail else ""))
    results.append((label, ok, detail))


def run_check(label: str, fn):
    try:
        fn()
        check(label, True)
    except Exception as e:
        check(label, False, str(e)[:120])


# ------------------------------------------------------------------------------- 1. imports
print("\n=== 1. Package imports ===")
PACKAGES = [
    "jobintel.db.models",
    "jobintel.ir.index",
    "jobintel.rec.resume",
    "jobintel.rec.hybrid",
    "jobintel.rec.store",
    "jobintel.rec.skillgap",
    "jobintel.web",
    "jobintel.web.backend",
    "jobintel.web.views.resume",
    "jobintel.web.views.recommend",
    "jobintel.web.views.market",
    "jobintel.web.views.skillgap",
]
for pkg in PACKAGES:
    try:
        importlib.import_module(pkg)
        check(f"import {pkg}", True)
    except Exception as e:
        check(f"import {pkg}", False, str(e)[:100])


# ------------------------------------------------------------------------------- 2. key data files
print("\n=== 2. Data files ===")
FILES = [
    ("index/job_index.pkl", "IR search index (run scripts/build_index.py to create)"),
    ("processed/skill_lexicon.json", "skill lexicon (run scripts/build_lexicon.py or ingest_all to create)"),
]
for rel_path, hint in FILES:
    path = DATA_DIR / rel_path
    alt_path = DATA_DIR / Path(rel_path).name
    ok = (path.exists() and path.stat().st_size > 0) or (alt_path.exists() and alt_path.stat().st_size > 0)
    check(f"data/{rel_path}", ok, "" if ok else f"missing — {hint}")


# ------------------------------------------------------------------------------- 3. database
print("\n=== 3. Database ===")
try:
    from sqlalchemy import create_engine, inspect, text
    from jobintel.db import models as M  # noqa: F401 — ensures metadata is populated
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    check("DB connects", True)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    required = {"users", "jobs", "skills", "user_skills", "job_interactions"}
    missing = required - tables
    check("required tables exist", not missing, ", ".join(sorted(missing)) if missing else "")
except Exception as e:
    check("DB connects", False, str(e)[:120])
    check("required tables exist", False, "DB not reachable")


# ------------------------------------------------------------------------------- 4. web routes & real content
print("\n=== 4. Web routes (test client & content assertions) ===")
try:
    from sqlalchemy import create_engine as _ce
    from jobintel.web.settings import WebSettings
    from jobintel.web.backend import Backend
    from jobintel.web import create_app

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        st = WebSettings(testing=True, secret_key="verify-key", warmup=False, data_dir=DATA_DIR)
        db_engine = _ce(DB_URL)
        be = Backend(st, db=db_engine)
        app = create_app(st, be)
        c = app.test_client()

        PUBLIC_ROUTES = [
            ("/", 200, ["Browse by job area", "Understand the job market"]),
            ("/jobs", 200, ['name="q"', "Find Jobs"]),
            ("/about", 200, ["Jobintel", "About"]),
            ("/privacy", 200, ["Privacy"]),
            ("/terms", 200, ["Terms"]),
            ("/health", 200, ["System status", "Model and data files"]),
            ("/login", 200, ["Log in", 'name="identifier"']),
            ("/register", 200, ["Create an account", 'name="password"']),
            ("/recommendations", 200, ["Find recommendations", "Number of results"]),
            ("/resume", 200, ["Resume Analyzer", 'name="resume"']),
            ("/career-insights", 200, ["Career Insights"]),
            ("/skill-gap", 200, ["Skill Gap Analysis", "Analyse gap"]),
        ]
        for route, expected, content_markers in PUBLIC_ROUTES:
            r = c.get(route)
            html = r.data.decode("utf-8", errors="replace")
            ok_status = (r.status_code == expected)
            has_markers = all(m.lower() in html.lower() for m in content_markers)
            no_traceback = ("Traceback" not in html) and ("500 Internal" not in html)
            ok = ok_status and has_markers and no_traceback
            detail = ""
            if not ok_status:
                detail = f"status {r.status_code} (expected {expected})"
            elif not has_markers:
                missing = [m for m in content_markers if m.lower() not in html.lower()]
                detail = f"missing markers: {missing}"
            elif not no_traceback:
                detail = "unhandled error/traceback present"
            check(f"GET {route} (verified content)", ok, detail)

        # Resume upload: unsupported extension must return 400
        r = c.post(
            "/resume/analyze",
            data={"resume": (io.BytesIO(b"not exe"), "resume.exe")},
            content_type="multipart/form-data",
        )
        check("upload .exe → 400", r.status_code == 400)

        # Auth redirect on protected routes
        for path in ("/profile", "/account", "/dashboard", "/saved"):
            r = c.get(path)
            ok = r.status_code == 302 and "/login" in r.headers.get("Location", "")
            check(f"GET {path} → redirect to login", ok)

except Exception:
    check("web route test", False, traceback.format_exc().splitlines()[-1])


# ------------------------------------------------------------------------------- 5. compile check
print("\n=== 5. Bytecode compilation ===")
import compileall
ok = compileall.compile_dir(str(ROOT / "jobintel"), quiet=2, force=False)
check("python -m compileall jobintel", bool(ok))


# ------------------------------------------------------------------------------- summary
print()
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed
print(f"=== Summary: {passed}/{total} checks passed ===")
if failed:
    print(f"  {failed} FAILED:")
    for label, ok, detail in results:
        if not ok:
            print(f"    - {label}" + (f": {detail}" if detail else ""))
    sys.exit(1)
else:
    print("  All checks passed.")
    sys.exit(0)
