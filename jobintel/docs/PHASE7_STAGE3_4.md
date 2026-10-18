# Phase 7 — Stages 3 and 4 implementation notes

Date: 2026-10-13

## Implemented

### Stage 3 — Resume Analyzer
- Added `/resume/analyze`, `/resume/save`, and `/resume/clear`.
- Uploads are size-limited by the existing Flask configuration, allow-listed by extension, checked against basic file signatures/structure, and processed in a temporary directory that is removed after analysis.
- PDF input is capped at 20 pages and encrypted PDFs are rejected. DOCX archive expansion and image dimensions have explicit limits.
- Uses the existing Phase 6 `read_document` and `ResumeAnalyzer`; it does not introduce a replacement parser or train a model.
- Displays extracted fields, skills, evidence, sections and warnings. OCR provenance is shown as a warning.
- The analysed resume stays in the existing in-memory, expiring `EphemeralStore`; saving is a separate authenticated action.
- Saving updates resume text/name and fills missing profile fields only; existing manually entered skills and profile preferences are preserved. Unknown skills are not inserted into the skills table.

### Stage 3 — Recommendations
- Re-enabled the earlier recommender flow using the current dark design and the existing Phase 6 `HybridRecommender` / `UserProfile` / `DLSimilarity` modules.
- Supports a typed profile, current-session resume, or saved profile; shows matches, missing skills, warnings and inactive signals.
- Keeps weights configurable through existing defaults and does not claim they are optimal.
- Signed-in users can explicitly save the current results. Saved results are user-scoped in the existing `recommendations` table.

### Stage 4 — Career Insights
- `/career-insights` reports real searchable-document counts by job area, grouped role-family counts, and skill frequency from the current index.
- Area filters use the same index cluster labels as Find Jobs.
- Related occupations are shown only when the existing knowledge graph returns them.
- Copy explains that counts describe the stored job snapshot, not the whole live market.

### Stage 4 — Skill Gap
- `/skill-gap` accepts a real target job or target role and skills from the current resume session, saved profile, or typed form.
- Uses the existing `analyze_gap` heuristic and `ItemCatalog`; displays matching/missing skills, market frequency, related held skills, occupation graph information and warnings.
- For signed-in users targeting a specific job, the current rows for that user/job are replaced with the latest report in the existing `skill_gaps` table.
- The page explicitly says importance is not a hiring probability and does not invent courses.

### Cross-cutting persistence
- The profile checklist now points to the implemented Resume Analyzer.
- The workspace and Account export wording reflect recommendation and skill-gap support.
- `/account/export` now includes user-scoped saved recommendation and skill-gap rows. The existing explicit account deletion routine already removes both tables.
- Added Stage 3–4 route/workflow tests and updated the public-page tests to expect real pages rather than placeholders.

## Verification status — be precise

- `python -m compileall -q jobintel tests`: passed.
- Jinja parsing of the four new templates (`recommend.html`, `resume.html`, `market.html`, `skillgap.html`): passed. The standalone Jinja check does not register the Flask-only custom filters used by some older templates; those reported filter errors are expected outside Flask's UI setup.
- `tests/test_rec_catalog_kg_gap.py`: 9 passed.
- `tests/test_rec_resume.py`: 13 passed.
- `tests/test_rec_documents.py` combined with other tests did not finish within the available timeout; do not count it as passed.
- Web tests were **not run** because Flask/Werkzeug are absent in this execution environment and package installation cannot reach the package index. Stage 3–4 web tests are present but unverified here. Run `python -m pip install -r requirements.txt` on the project machine, then `python -m pytest tests/test_web_*.py -q`.
- No MariaDB-backed Stage 3–4 verification, browser screenshots, keyboard audit or full upload hostile-input audit has been performed here.

## Dependencies and limitations

- `pypdf` is now listed for PDF text extraction.
- The existing reader optionally imports PyMuPDF (`fitz`) for scanned-PDF rendering. It is intentionally **not added as a dependency here** because the project handoff flags its AGPL licensing choice as unresolved. Without that optional dependency, a scanned PDF without a text layer may fail with a clear message. Image OCR still depends on the Tesseract binary and the existing OCR libraries.
- Stage 5 is still required: full web tests on SQLite and MariaDB, upload abuse/size/page-limit tests, authentication and cross-user persistence checks, browser/responsive screenshots, keyboard/contrast checks, verification script, final handoff and packaging.
- Phase 6 merge remains unapplied. No new recommendation/DL engine or schema was introduced; the pages use the existing Phase 6 modules and Phase 1 tables.
