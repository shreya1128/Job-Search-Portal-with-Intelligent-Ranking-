# Phase 6 - resume analysis, recommenders, skill gap, hybrid ranking

New package `jobintel/rec/` + scripts + tests; **no Phase 1-5 file is modified except `README.md` / `requirements.txt` (Phase 6 sections / dependencies added)**.  235 tests pass (170 Phase 1-5 + 65 Phase 6 / 5b tooling).
Phase 6 was built with the project owner's go-ahead before the Phase 5 gate had been passed; Phase 5 has since been verified and its limitations addressed (`docs/PHASE5B_MCF.md`).

## Requirement checklist
| Requirement | Status | Evidence / limit |
|---|---|---|
| Content-based recommendation | COMPLETE | `rec/catalog.py` (idf-weighted skill cosine + requirement coverage) |
| Knowledge-based recommendation, role/skill reasoning, cold start | COMPLETE | `rec/knowledge.py`; works from a preferred role alone (tested); graph = ESCO + curated (no O*NET in the rebuilt database) |
| Collaborative filtering (viewed / clicked / saved / applied) | COMPLETE as machinery; **SYNTHETIC data only** | `rec/interactions.py`; the project has no real interaction data |
| Skill-gap analysis (matching / missing, importance, market frequency) | COMPLETE | `rec/skillgap.py`; importance is a stated, unvalidated heuristic |
| Hybrid ranking, 11 signals, normalised, configurable | COMPLETE | `rec/hybrid.py`; default weights provisional, not claimed optimal |
| DL representation / similarity signal | COMPLETE (skill space) | `rec/dlsim.py::DLSkillSimilarity` + Phase 5b v2 embeddings (59,748 jobs, all sources); inactive-with-reason when artefacts are missing |
| Resume information extraction | COMPLETE | text, DOCX, PDF (text layer or scanned), images: `rec/documents.py`, `rec/resume.py` |
| NER | COMPLETE (rule-based) | EMAIL, PHONE, URL, PERSON (heuristic), ORG, DATE_RANGE + skills / degrees / certifications / roles / locations; measured on SYNTHETIC resumes only; no trained model |
| Resume image processing, region detection, OCR | COMPLETE (classical CV + Tesseract) | deskew, binarise, region detection, column-aware reading order, per-word confidence; measured on SYNTHETIC renders only - no real scans were available |
| Recommendation evaluation: Precision@K, Recall@K, coverage, hit rate; different weight combinations | COMPLETE with stated limits | synthetic CF hold-out; weight combinations re-ranked on the AI-judged pool |
| Persistence in the Phase 1 tables | COMPLETE | tested on SQLite and run against MariaDB (`analyze_resume.py --save-user`) |
| Results page / UI | Phase 7 | - |

## Run
```
python scripts/analyze_resume.py --file data/samples/sample_resume_scan.png --role "data analyst" --city Pune --strict-city --k 8 --gap 2 [--save-user NAME] [--json out.json]
python scripts/generate_synthetic_interactions.py --users 300 --seed 42 [--write-db]      # SYNTHETIC
python scripts/evaluate_recommenders.py --users 300 --seed 42 --k 10                       # SYNTHETIC hold-out
python scripts/evaluate_resume_pipeline.py --n-text 200 --n-ocr 12 --seed 11               # SYNTHETIC resumes / renders
python scripts/evaluate_hybrid_weights.py --judgments data/eval/relevance_judgments_assistant_labeled.xlsx
```
`data/samples/sample_resume.txt` and `sample_resume_scan.png` are FICTIONAL (the scan is a synthetic render with 2 degrees of rotation and noise).

## Measured results (every table names its data)
### 1. Resume fields from clean text - SYNTHETIC resumes, held-out seed 11, n = 200
Email P/R 1.00/1.00, phone 1.00/1.00, URL 1.00/1.00, organisations 1.00/1.00, certifications 1.00/1.00, name exact match 1.00; all three sections found 1.00.
Skills: listed-in-skills-section P/R 1.00/1.00; all extracted skills P 0.71 / R 1.00 (the extra "skills" are job titles, degrees and certificate names that the lexicon treats as skills, e.g. *data analyst*, *information technology*; the analyser gives them a lower evidence weight than listed skills).
Experience from date ranges: mean absolute error 0.02 years for month-precision ranges, **0.49 years for year-only ranges** (e.g. "2018 - 2021" cannot be more precise); stated experience matched exactly in 1.00 of the 95 resumes that state it.
**Development vs held-out:** the organisation rules were fixed after error analysis on seed 7 (four defects: names split at "Institute of ...", matches across line breaks, degrees such as "Information Technology" and certificates read as organisations); the numbers above are from the untouched seed 11.  A fixed pool of fictional names makes this an easy template: real resumes will score lower.

### 2. Scanned-resume round trip - SYNTHETIC renders, 12 resumes x 7 variants
| variant | CER mean | CER max | OCR confidence | skew corrected (deg) | email recall | org recall | listed-skill recall | s / page |
|---|---|---|---|---|---|---|---|---|
| clean | 0.0004 | 0.0025 | 94 | -0.0 | 0.917 | 1.000 | 1.000 | 2.4 |
| rotated_3deg | 0.0003 | 0.0017 | 94 | -3.0 | 0.917 | 1.000 | 1.000 | 2.5 |
| noise_sigma25 | 0.0007 | 0.0025 | 93 | -0.0 | 0.833 | 1.000 | 1.000 | 3.3 |
| low_res_50pct | 0.0001 | 0.0016 | 94 | -0.0 | 0.917 | 1.000 | 1.000 | 2.9 |
| blur_jpeg40 | 0.0000 | 0.0000 | 94 | -0.0 | 1.000 | 1.000 | 1.000 | 2.3 |
| two_columns | 0.0019 | 0.0180 | 94 | -0.0 | 0.917 | 0.906 | 1.000 | 5.0 |
| two_columns_rot-2 | 0.0016 | 0.0180 | 94 | +2.0 | 0.917 | 0.906 | 1.000 | 4.9 |
CER = character error rate against the rendered text.  Region detection + column-aware reading order is only used when a multi-column page is detected (whole-page OCR is more accurate on single columns but interleaves two columns: CER 0.60 vs 0.002 on the tuning page).  **Even at CER 0.0004 an exact-match field such as an e-mail address can be wrong (email recall 0.917 on clean renders): OCR provenance and a "verify the extracted fields" warning are attached to every profile read by OCR.**  Design choices (deskew, automatic layout, grayscale crops) were made on separate tuning seeds (1000+), not on the evaluation seeds.

### 3. Collaborative filtering - SYNTHETIC interactions, 300 synthetic users, 60,808-job catalogue, k = 10
| system | hit@10 = recall@10 (95% CI) | precision@10 | catalogue coverage |
|---|---|---|---|
| item-based CF | 0.083 (0.057-0.120) | 0.0083 | 0.0123 |
| popularity | 0.000 (0.000-0.013) | 0.0000 | 0.0003 |
| persona oracle (sees the generator's hidden skills) | 0.557 (0.500-0.612) | 0.0557 | 0.0467 |
Random expectation 0.00017.  This shows the machinery recovers the structure the generator put in; it says nothing about real users.

### 4. Weight combinations of the hybrid - labels are AI-judged (`assistant_judged`), 20 queries, judged-pool re-ranking, DL signal ON
Full table (all single signals, leave-one-out, equal weights): `docs/phase6_weight_evaluation.md`.
| configuration | nDCG@10 | P@10 (grade 2) | nDCG@10 minus default [95% CI] | Holm-adj. p |
|---|---|---|---|---|
| default hybrid | 0.864 | 0.720 | - | - |
| default without dl | 0.846 | 0.705 | -0.018 [-0.035, -0.002] | 0.82 |
| only ir | 0.852 | 0.700 | -0.012 [-0.043, +0.020] | 1.00 |
| IR family only (ir, lm, dfr, passage, pagerank) | 0.818 | 0.720 | -0.046 [-0.133, +0.009] | 1.00 |
| recommender signals only (content, knowledge, skill_match, preference, dl) | 0.839 | 0.725 | -0.026 [-0.067, +0.005] | 1.00 |
| equal weights (all active) | 0.861 | 0.715 | -0.003 [-0.021, +0.014] | 1.00 |
| only knowledge | 0.628 | 0.505 | -0.236 [-0.323, -0.161] | 0.00 |
| random order of the pool | 0.716 | 0.574 | -0.148 [-0.204, -0.095] | 0.00 |
**What is and is not supported:** no reasonable weighting is distinguishable from the default after Holm correction; removing the DL signal costs 0.018 nDCG (nominal interval excludes 0, adjusted p 0.82: suggestive, not established); the knowledge signal ALONE ranks the judged pool worse than random order (adjusted p 0.001) although it does no harm at its 10% weight; "preference" exists for only 2 of 20 queries (profiles come from the query text, which rarely names a city or experience).  Profiles are derived from the query text, candidates are the judged jobs (IR-biased pool), most pooled jobs are relevant at grade >= 1 so lenient P@10 is near its ceiling.  Default weights stay provisional.

## Faults found while building / testing (and what was done)
* Location was one soft signal of 11, so a Pune user got no Pune job in the top 8: added `strict_location` and a warning when no top result is in the preferred city (weights NOT re-tuned).
* The Phase 1 certification pattern spans line breaks / headings on resumes: handled in the resume layer (Phase 1 extractor untouched).
* Reported weights were rounded (1e-6) and contributions were float32 (2e-9): the explanation now sums exactly to the score.
* Organisation rules (4 defects, above); the synthetic generator's month convention (exclusive vs inclusive end month) corrected before any measurement.
* Evaluation-script design flaws caught on the first run: signals active for only some queries were dropped for all; ties were broken by pool order, which IS the default ranking.
* The Phase 5 embedding builder maps skills by `skill_id`, which is not portable between databases; the new scripts map by name / by (source, external_id).
* The sandbox restarted during training and killed the background job; the resumable design (skip trained models) recovered it with no lost work.

## Known limits
* No real relevance labels, interaction logs, user profiles or scanned resumes exist for this project: every recommender / resume number above is on synthetic or AI-judged data.
* The rebuilt database lacks O*NET; ESCO `related_to` edges are loose, so skill-gap "bridges" are suggestions.
* `ir` already blends lm / dfr / passage / pagerank; the separate small weights double-count by design.
* Salary preference is not implemented; Tesseract language is English only; handwriting, icons, tables and multi-page layouts are untested.
* The skill-space DL signal is only as good as the denoising autoencoder (recall@10 0.405 vs 0.348 co-occurrence in v2, single seed).

## Merging
Unzip `jobintel_phase6_additions.zip` over the project root (it contains changed `README.md` / `requirements.txt` / `docs/HANDOFF.md`, so back those up if you edited them) and `jobintel_phase5b_models.zip` for the v2 models / embeddings.  Install the Tesseract program for OCR.
