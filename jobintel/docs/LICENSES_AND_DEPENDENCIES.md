# JobIntel - Software Licenses and Dependency Audit

This document details the licensing terms of the JobIntel source code, third-party libraries, and external datasets.

---

## 1. Primary Codebase License

The JobIntel core codebase is distributed under the permissive **MIT License** (see [LICENSE](file:///C:/Users/shrey/OneDrive/Documents/Desktop/CIE/jobintel/LICENSE)).

---

## 2. AGPL Elimination and PDF Engine Audit

In previous revisions, PyMuPDF (`fitz`) was considered for rasterising scanned PDFs. Because PyMuPDF is licensed under the strict **GNU Affero General Public License (AGPL v3)**, using it would impose copyleft constraints across the entire web application and backend.

### Resolution:
- **PyMuPDF (`fitz`) has been completely excised** from all production code paths (`jobintel/rec/documents.py`, `jobintel/rec/ocr.py`, `upload_guard.py`).
- **Standard Permissive Stack Adopted**:
  - `pypdfium2`: **Apache-2.0 / BSD-3-Clause** (permissive C++ PDFium bindings).
  - `pypdf`: **BSD-3-Clause** (pure Python text layer and metadata extractor).
  - `pdfplumber`: **MIT License**.
  - `python-docx`: **MIT License**.
  - `pytesseract`: **Apache-2.0**.
  - `reportlab`: **BSD License** (used in test suite for fixture PDF generation).

---

## 3. Dependency License Inventory

| Package | Version | Primary License | Permissiveness / Category |
|---|---|---|---|
| `Flask` | 3.0+ | BSD-3-Clause | Permissive web framework |
| `Werkzeug` | 3.0+ | BSD-3-Clause | WSGI utility library |
| `Jinja2` | 3.1+ | BSD-3-Clause | Templating engine |
| `SQLAlchemy` | 2.0+ | MIT | Database abstraction layer |
| `PyMySQL` | 1.1+ | MIT | Pure Python MySQL driver |
| `waitress` | 3.0+ | ZPL 2.1 | Permissive WSGI production server |
| `pandas` | 2.0+ | BSD-3-Clause | Data analytics and manipulation |
| `numpy` | 1.26+ | BSD-3-Clause | Numerical vector computing |
| `scipy` | 1.11+ | BSD-3-Clause | Scientific computing / sparse graphs |
| `scikit-learn` | 1.3+ | BSD-3-Clause | Machine learning algorithms |
| `tensorflow-cpu` | 2.16+ | Apache-2.0 | Deep learning representation models |
| `nltk` | 3.8+ | Apache-2.0 | Natural language stemming |
| `beautifulsoup4`| 4.12+ | MIT | HTML parser and sanitizer |
| `openpyxl` | 3.1+ | MIT | Excel reading for O*NET and judgments |
| `cryptography` | 41.0+ | Apache-2.0 / BSD | MySQL 8 caching_sha2 authentication |
| `python-dotenv` | 1.0+ | BSD-3-Clause | Environment configuration parser |
| `pytest` | 8.0+ | MIT | Unit and integration testing |

All direct runtime dependencies use standard OSI-approved permissive licenses (**MIT**, **BSD**, or **Apache 2.0**).

---

## 4. Dataset Licenses and Attribution

| Dataset | Provider / Source | License / Terms of Use | Usage in JobIntel |
|---|---|---|---|
| **ESCO v1.2** | European Commission | Open Data (ODC-By / CC-BY 4.0) | Taxonomy of skills and occupations |
| **O\*NET 29.0** | US Department of Labor | CC BY 4.0 | Knowledge relationships & technology skills |
| **MyCareersFuture (MCF)**| Singapore Govt (data.gov.sg) | Singapore Open Data License | 1,000 live jobs (salaries, skills, experience) |
| **EMSCAD** | Univ of the Aegean / Kaggle | Research CC0 / Open Data | 1,000 real job postings + fraud evaluation |
| **Naukri Data Science** | Kaggle | CC0: Public Domain | 1,000 tech & analytics job descriptions |
| **LinkedIn Sample** | Kaggle | CC0 / Attribution | 1,000 job postings with metadata & skills |
| **Resume Dataset** | Kaggle / snehaanbhawal | CC0: Public Domain | 2,484 multi-domain resume texts |
| **Synthetic Interactions** | Generative model (`scripts/generate_synthetic_interactions.py`) | MIT (Project Generated) | **Strictly labeled `is_synthetic = 1`** |
