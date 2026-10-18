# JobIntel — Web Application Guide

> MCA Academic Project · IR + DL subjects · Phase 7 (Web Application)

---

## Quick-start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 – 3.12 |
| MySQL **or** SQLite | MySQL 8+ recommended; SQLite works for local/demo |
| Tesseract OCR | Optional — enables image/scanned-PDF resume upload |

All Python dependencies are in `requirements.txt`.  Install them once:

```powershell
pip install -r requirements.txt
```

---

### 1 · Configure environment

Create a `.env` file in the project root (copy from `.env.example` if present):

```dotenv
# Minimum required settings
JOBINTEL_SECRET_KEY=change-me-to-a-long-random-string
JOBINTEL_DATA_DIR=data                # folder containing job_index.pkl, skill_lex.json, etc.

# Database (leave blank to use SQLite at jobintel.db)
JOBINTEL_DB_URL=mysql+pymysql://user:password@localhost/jobintel

# Optional
JOBINTEL_MAX_UPLOAD_MB=8             # default 8
JOBINTEL_DEBUG=0                     # set 1 to show tracebacks in /health
```

---

### 2 · Initialise the database

```powershell
python scripts/init_db.py
```

> **Note**: This creates all tables.  Run it once; re-running is safe (it uses `CREATE TABLE IF NOT EXISTS`).

---

### 3 · Build the search index (one-time)

The IR search index must exist before the job-search and recommendations pages work.
It is built from the jobs already in the database:

```powershell
python scripts/build_index.py --write-db
```

The index is written to `data/job_index.pkl`.

---

### 4 · Run the app

```powershell
flask --app jobintel.web run
```

Open **http://localhost:5000** in a browser.

For demo/presentation, disable reloader to avoid double-startup:

```powershell
flask --app jobintel.web run --no-debugger --no-reload
```

---

### 5 · Verify the installation

```powershell
python scripts/verify_phase7.py
```

Prints one `OK / FAIL` line per check:
- Python package imports
- Required data files
- Database connectivity and schema
- All 12 public routes return 200
- Upload endpoint rejects invalid files
- Bytecode compiles cleanly

Exit code 0 = all checks passed.

---

## Pages and features

| URL | What it does |
|---|---|
| `/` | Home — job areas, example roles, quick links |
| `/jobs` | Full-text search with filters (area, city, experience), pagination |
| `/job/<id>` | Job detail, skills, apply link (http/https only), related jobs |
| `/recommendations` | 11-signal hybrid recommender — use form, analysed resume, or saved profile |
| `/resume` | Upload and analyse a resume; extracts skills, experience, education |
| `/career-insights` | Market skill frequency, popular roles, job-area breakdown |
| `/skill-gap` | Compare your skills against a specific job or target role |
| `/profile` | Edit name, education, target role, skills, experience (login required) |
| `/account` | Change password, download your data as JSON (login required) |
| `/dashboard` | Saved jobs, applied jobs, saved recommendations (login required) |
| `/register`, `/login` | Account creation and authentication |
| `/health` | System status — which backends are ready |

---

## Supported resume formats

The Resume Analyzer (`/resume`) accepts:

| Format | How it's read |
|---|---|
| `.txt`, `.md` | Direct UTF-8 read |
| `.pdf` | `pypdfium2` (rendered pages) or `pypdf` (text layer); scanned pages use OCR if Tesseract is installed |
| `.docx` | `python-docx` |
| `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp` | Pillow validation + Tesseract OCR (requires Tesseract) |

**Limits**: 8 MB file size, 20 pages for PDFs, 120,000 characters of extracted text.

Password-protected PDFs are rejected.

---

## Known limitations

| Limitation | Reason |
|---|---|
| Recommendations require a pre-built index | Run `build_index.py` first |
| DL similarity signal requires trained model files | Run `scripts/train_autoencoder_v2.py` to produce them |
| Collaborative filtering requires interaction data | Needs real user interactions in the DB; falls back gracefully |
| OCR requires Tesseract | Not installed by default; image/scanned PDF pages are skipped if absent |
| MyCareersFuture data not included | Data was unavailable at submission time; the system works with the seeded dataset |
| Skill-gap KG links unavailable without graph | Knowledge-graph paths are skipped when the KG is not loaded; basic gap analysis still runs |

---

## Running the test suite

```powershell
# All web tests (Stages 2, 3, 4)
python -m pytest tests/test_web_account.py tests/test_web_helpers.py tests/test_web_pages.py tests/test_web_stage2.py tests/test_web_stage34.py -v

# Phase 6 recommender unit tests
python -m pytest tests/test_rec_catalog_kg_gap.py tests/test_rec_dlsim.py tests/test_rec_documents.py tests/test_rec_hybrid.py tests/test_rec_interactions.py tests/test_rec_resume.py tests/test_rec_store.py tests/test_rec_upload_guard.py -v

# Everything fast (no training)
python -m pytest tests/ -v --ignore=tests/test_mapreduce.py
```

> Tests use an in-memory SQLite database and a seeded 30-job fixture.  No live MySQL, no TensorFlow model files, no Tesseract required to run them.

---

## Project structure (Phase 7 additions)

```
jobintel/
  web/
    views/
      resume.py        # Resume upload / analyse / save / clear routes
      recommend.py     # 11-signal hybrid recommender
      market.py        # Career Insights aggregates
      skillgap.py      # Skill-gap analysis vs job or role
      _profile.py      # Shared profile helpers (load, save, export, normalize)
    templates/
      resume.html
      recommend.html
      market.html
      skillgap.html
    static/css/
      app.css          # Extended with Stage 3/4 component styles
  rec/
    documents.py       # PDF / DOCX / image reading (pypdfium2 primary)
    hybrid.py          # HybridRecommender (11 signals)
    resume.py          # ResumeAnalyzer (NLP extraction)
    skillgap.py        # analyze_gap()
    upload_guard.py    # Extension/content mismatch detection
scripts/
  verify_phase7.py     # End-to-end health check
docs/
  WEB_APP.md           # This file
  PHASE7_STAGE3_4.md   # Stage 3/4 implementation notes
```

---

## Demonstration checklist

Use this checklist during the presentation / evaluation:

- [ ] `python scripts/verify_phase7.py` exits 0
- [ ] Home page loads; shows job areas and "Understand the job market"
- [ ] Job search: `q=data analyst` → results with "Why it matched"
- [ ] Job detail: skills listed, apply link present (or clearly absent)
- [ ] Resume Analyzer: upload a `.txt` resume → extracted skills shown
- [ ] Career Insights: `/career-insights` → skill frequency table visible
- [ ] Skill Gap: `/skill-gap` → select a job, submit → skill-by-skill comparison
- [ ] Recommendations: `/recommendations?run=1&src=form&role=data+analyst&skills=python,sql&k=5` → results or graceful "unavailable"
- [ ] Register, log in, save a skill to profile, visit dashboard
- [ ] Upload `.exe` → rejected with 400 and user-readable error
- [ ] `/health` → JSON listing which backends are ready vs unavailable

---

*Last updated: Phase 7 Stage 5 — 14 October 2026.*
