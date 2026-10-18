# JobIntel — Final Project Handoff & Architecture Guide

> **MCA Final Project** · Information Retrieval (IR) + Deep Learning (DL)  
> **Phase**: Complete (Phases 1–8 Production Deliverable)  
> **Date**: 14 October 2026  
> **Status**: Demonstrable, Fully Tested, Hardened & Packaged

---

## 1. Project Overview & Architecture

**JobIntel** is an end-to-end Job Market Intelligence and Hybrid Job Recommendation System. It integrates probabilistic information retrieval engines, deep learning sequence representations, knowledge graphs (ESCO & O\*NET), collaborative filtering, and an automated resume parser into a secure multi-page web application.

### High-Level Architecture

```
[ Crawled & Ingested Postings / Resumes ]
                    │
                    ▼
       ┌────────────────────────┐
       │   Extraction & Lexicon │  (SkillLexicon, Gazetteers, Regex/Spacy)
       └────────────┬───────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
┌─────────────────┐   ┌───────────────────────────┐
│ Database Models │   │ IR Inverted Index         │
│ (MySQL/SQLite)  │   │ - TF-IDF / BM25           │
│ 21 Relations    │   │ - Language Model Dirichlet│
└────────┬────────┘   │ - DFR (InL2) & PageRank   │
         │            │ - Passage-Level Scoring   │
         │            └─────────────┬─────────────┘
         │                          │
         └─────────────┬────────────┘
                       ▼
┌────────────────────────────────────────────────────────┐
│ Hybrid Recommendation Engine (11 Scoring Signals)      │
│ 1. BM25/TF-IDF  2. LM Dirichlet  3. DFR InL2            │
│ 4. Passage      5. PageRank      6. DL Similarity       │
│ 7. Content      8. Collab Filter 9. Knowledge Graph     │
│ 10. Skill Match 11. Profile Preferences                │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ Flask Web Application (Phase 7 Stages 1–5 & Phase 8)   │
│ - Home & Job Search (/jobs)                            │
│ - 11-Signal Recommender (/recommendations)             │
│ - Secure Resume Analyzer (/resume)                     │
│ - Career Insights (/career-insights)                   │
│ - Skill Gap Analysis (/skill-gap)                      │
│ - User Dashboard & Profiles (/dashboard, /profile)     │
│ - System Health (/health)                              │
└────────────────────────────────────────────────────────┘
```

---

## 2. Implemented Features & Status

| Module / Feature | Description | Status |
|---|---|---|
| **Real Datasets Ingested** | ESCO v1.2, O\*NET 29.0, MyCareersFuture (MCF), EMSCAD, Naukri, LinkedIn, Resumes. Synthetic jobs & interactions strictly tagged `is_synthetic = 1`. | ✅ **Complete & Verified (4,300 jobs, 17k skills, 4k occs)** |
| **IR Engine** | Multi-model search (BM25, Language Models, DFR InL2, Passage retrieval, PageRank, Distributed Engine) | ✅ **Complete & Verified (3,951 docs, 279k postings)** |
| **Deep Learning** | RNN, BiRNN, LSTM, GRU architectures + Denoising Autoencoder for skill representations | ✅ **Trained models included** |
| **Recommender Core** | 11-signal hybrid scoring with graceful fallback when signals are absent | ✅ **Tested & Active** |
| **Resume Analyzer** | Multi-format parser (PDF via pypdfium2/pypdf, DOCX, TXT) with upload guard, OCR budgets (5 pages, 15s deadline), no AGPL | ✅ **Hardened & Active** |
| **Career Insights** | Aggregated market demand, salary distributions, skill frequencies, popular roles | ✅ **Active** |
| **Skill Gap Analyzer** | Gap analysis matching user skills against targeted jobs (alphabetical job picker) or roles with ESCO/O\*NET benchmarks | ✅ **Active** |
| **Web Authentication** | CSRF protection, secure passwords, session store, user profile editing, account export | ✅ **Active** |
| **Security & Hardening** | Strict CSP, X-Frame-Options: DENY, X-Content-Type-Options: nosniff, session isolation, 404 sanitization | ✅ **Verified (166 web tests passed)** |
| **Production WSGI** | Native multi-threaded Waitress WSGI runner (`run_production.py`) and standard `wsgi.py` entrypoint | ✅ **Active** |

---

## 3. Setup & Installation

### Requirements
- Python 3.10 – 3.12 (Tested on Python 3.12.10 Windows x64)
- MySQL 8.0+ or SQLite 3 (SQLite is ready out of the box in `jobintel.db`)
- Tesseract OCR (Optional: only needed for image/scanned OCR)

### Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 4. Database Initialization & Index Commands

The packaged project already contains the complete database in `jobintel.db` and index files in `data/index/job_index.pkl`. If you ever need to rebuild them from scratch:

```powershell
# 1. Initialize tables (creates all 21 schema tables)
python scripts/init_db.py

# 2. Ingest real datasets (ESCO, O*NET, MCF, EMSCAD, Naukri, LinkedIn, Resumes)
python scripts/ingest_all.py --rebuild-lexicon --only knowledge naukri emscad mcf linkedin resumes gazetteer --limit 1000

# 3. Compute PageRank graph scores
python scripts/compute_pagerank.py

# 4. Build IR Search Index and passages
python scripts/build_index.py --write-db

# 5. Generate synthetic interactions for recommender evaluation (strictly flagged is_synthetic = 1)
python scripts/generate_synthetic_interactions.py --users 300 --write-db
```

---

## 5. How to Run the Web Application

### Option A: Development Runner
```powershell
python scripts/run_web.py --port 5000
```

### Option B: Production WSGI Runner (Waitress)
```powershell
python scripts/run_production.py --host 127.0.0.1 --port 5000 --threads 4
```

Open **http://localhost:5000** in your browser.

### Key Routes to Demo
- **Home**: `http://localhost:5000/`
- **Job Search**: `http://localhost:5000/jobs?q=data+analyst`
- **Recommendations**: `http://localhost:5000/recommendations`
- **Resume Parser**: `http://localhost:5000/resume`
- **Career Insights**: `http://localhost:5000/career-insights`
- **Skill Gap**: `http://localhost:5000/skill-gap`
- **Health Diagnostics**: `http://localhost:5000/health`

---

## 6. How to Run Tests

```powershell
# Phase 7 Web test suite (166 tests across Stages 1–5)
python -m pytest tests/test_web_*.py -v

# Phase 6 Recommender test suite
python -m pytest -k "test_rec"

# End-to-end system verification
python -X utf8 scripts/verify_phase7.py

# Entire repository test suite (418 passed, 0 failed, 6 skipped)
python -m pytest tests/
```

---

## 7. Exact Final Test Results

Executed on the final packaged tree:

### Full Project Test Run
- **Total Tests Collected**: 425
- **Passed**: **418 passed** (100% of runnable tests)
- **Skipped**: **6 skipped** (Optional OCR test when Tesseract binary not on PATH; optional IPC tests)
- **Deselected**: 1 (Scrapy process test)
- **Failed**: **0 failed**
- **Errors**: **0 errors**
- **Execution Time**: 133.88s

### Web Suite Breakdown (`tests/test_web_*.py`)
- `tests/test_web_account.py`: **65 passed**
- `tests/test_web_helpers.py`: **32 passed**
- `tests/test_web_pages.py`: **22 passed**
- `tests/test_web_stage2.py`: **32 passed**
- `tests/test_web_stage34.py`: **10 passed**
- `tests/test_web_stage5.py`: **5 passed**
- **Total Web Tests**: **166 / 166 passed (100%)**

### End-to-End Verification (`scripts/verify_phase7.py`)
- **Package Imports**: 12/12 OK
- **Data & Index Files**: 2/2 OK (`data/index/job_index.pkl`, `data/processed/skill_lexicon.json`)
- **Database Connectivity & Schema**: 2/2 OK (21 tables verified)
- **Web Routes (All 12 public routes return 200 with verified content)**: 12/12 OK
- **Security & Upload Protection (.exe rejected with 400)**: 1/1 OK
- **Auth Redirects for Protected Pages**: 4/4 OK
- **Bytecode Compilation**: 1/1 OK
- **Overall**: **34 / 34 checks passed (100%)**

---

## 8. Artifacts Included vs Excluded

### Included in Package
- Complete source code (`jobintel/`, `scripts/`, `tests/`, `tools/`, `docs/`)
- Rebuilt search index: `data/index/job_index.pkl` (3,951 documents, 279,387 postings)
- Processed skill lexicon: `data/processed/skill_lexicon.json` (4,213 entries)
- Location gazetteer: `data/processed/location_gazetteer.json`
- Pre-populated SQLite database: `jobintel.db` (4,300 jobs, 17,442 skills, 4,075 occupations, 163k knowledge edges)
- Deep learning model artifacts:
  - `data/dl/models/ae_basic_v2.keras`
  - `data/dl/models/ae_deep_v2.keras`
  - `data/dl/models/ae_denoising_v2.keras` and `ae_denoising.keras`
  - Trained sequence models across multiple random seeds (`birnn_s42`..`s46`, `gru_s42`..`s46`, `lstm_s42`..`s46`, `rnn_s42`..`s46`)
  - Deep learning vocabularies and embeddings: `ae_vocab.json`, `ae_vocab_v2.json`, `tokenizer.json`, `embeddings_skills_v2.npz`
- Evaluation benchmarks & labeled judgments (`data/eval/`, `data/rec/`)
- Production runner `scripts/run_production.py` and `wsgi.py`
- Complete documentation: `PHASE8_REPORT.md`, `SYLLABUS_MAPPING.md`, `TESTING_AND_VERIFICATION.md`, `DEPLOYMENT_AND_SECURITY.md`, `LICENSES_AND_DEPENDENCIES.md`

### Excluded & Rationale
- `.env` / credentials: Excluded per security policy (use `.env.example`).
- `__pycache__` and `*.pyc`: Excluded to eliminate redundant binary cache bloat.
- `.pytest_cache`: Excluded temporary test artifacts.
- Older nested backup ZIPs in `_backups/`: Excluded to avoid recursive archive duplication.
