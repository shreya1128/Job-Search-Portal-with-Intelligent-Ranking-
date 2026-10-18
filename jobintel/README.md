# Job Market Skill Analysis and Job Recommendation System - Phases 1-5: data foundation, IR engine, parallel/distributed IR, IR evaluation, deep learning

Phases 1-2 of 8 for the MCA project (Information Retrieval + Deep Learning). **Phase 1** builds everything the later phases stand on (**Phase 2 - the IR engine - is described at the end of this file and in `docs/IR_ENGINE.md`**):
the MySQL schema, real-data ingestion, text preprocessing, the finite-automata information extractor, the skill/job knowledge base and
the web crawlers. **No search ranking, deep learning or web UI yet** - those are Phases 2-7.

## What Phase 1 delivers

| Area | What exists | Where |
|---|---|---|
| Database | 21 normalised MySQL tables (the 20 required + `occupations`), PK/FK/CHECK/UNIQUE/indexes, ER diagram, DDL | `jobintel/db/`, `docs/schema.sql`, `docs/er_diagram.mmd` |
| Ingestion | 4 real job datasets -> one schema, with original **and** processed text, de-duplication, ~65k jobs | `jobintel/ingest/` |
| Preprocessing (IR module 2) | HTML removal, mojibake repair, tokenising, stop-words, normalisation, Porter stemming, exact + SimHash duplicate detection, missing-field handling | `jobintel/processing/` |
| Finite automata (IR module 3) | NFA simulator + Aho-Corasick; experience / degree / salary / certification / role automata; state diagrams | `processing/fsa.py`, `patterns.py`, `docs/FINITE_AUTOMATA.md`, `docs/fsa/*.dot` |
| Information extraction | experience, education, salary, certifications, employment type, seniority, skills, locations, roles from free text | `jobintel/extraction/` |
| Knowledge base (semantic web) | curated skill/role graph + ESCO + O*NET -> 4,075 occupations, 27k skills, 164k relations | `ingest/knowledge.py`, `ingest/curated_kb.py` |
| Crawler (IR modules 18-20) | URL frontier -> polite fetcher (robots.txt) -> **BeautifulSoup** parser -> duplicate filter -> storage; the same crawler as a **Scrapy** spider; local fixture website; hyperlink graph for PageRank | `jobintel/crawler/` |
| Offline fallback | synthetic sample jobs + fixture site, so everything runs with no internet and no big datasets | `jobintel/sample_data.py` |
| Tests | 135 automated tests | `tests/` |

## Quick start

Python 3.10-3.12 (3.12 tested). MySQL 8 / MariaDB 10.5+ (tested on MariaDB 10.11; see `docs/DESIGN_NOTES.md`).

```powershell
# 1. environment
cd jobintel
py -m venv .venv ; .venv\Scripts\activate           # Linux/macOS: python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. database (run in the MySQL shell, choose your own password)
#    CREATE DATABASE jobintel CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
#    CREATE USER 'jobintel'@'localhost' IDENTIFIED BY 'your_password';
#    GRANT ALL ON jobintel.* TO 'jobintel'@'localhost';
copy .env.example .env                                # then edit DB_PASSWORD (and dataset paths, below)

# 3. datasets (see docs/DATASETS.md for where each one comes from)
python scripts/fetch_datasets.py                      # downloads MyCareersFuture + ESCO from GitHub
#    put/point the others: NAUKRI_CSV, EMSCAD_CSV, ONET_DIR, RESUMES_CSV, LINKEDIN_META_CSV in .env

# 4. create tables + load everything that is available (~5 min for the full data)
python scripts/init_db.py --export-docs
python scripts/ingest_all.py                          # or: --only knowledge naukri --limit 500   for a quick try
python scripts/verify_phase1.py                       # integrity + quality report -> docs/phase1_report.md

# 5. crawlers (offline, against the local fixture site)
python scripts/run_crawler.py --engine bs4 --ingest
python scripts/run_crawler.py --engine scrapy

# 6. tests
python -m pytest                                      # 134 fast tests (SQLite, no datasets needed; TensorFlow tests skip if it is not installed)
python -m pytest -m slow                              # + Scrapy-vs-BeautifulSoup equivalence
```

No datasets at all? `python scripts/ingest_all.py` automatically loads the **synthetic** offline sample so the system still runs.

## Project layout
```
jobintel/
  config.py            all paths / DB settings from environment (.env) - no secrets in code
  records.py           JobRecord: the one shape every source is converted to
  db/                  models.py (schema), session.py, export_schema.py
  processing/          text.py  fsa.py  patterns.py  dedupe.py  stemmer.py  stopwords.py
  extraction/          lexicon.py  seed_skills.py  build_lexicon.py  extractor.py
  ingest/              loaders.py  common.py  load_db.py  knowledge.py  curated_kb.py  resumes.py  gazetteer.py
  crawler/             frontier.py  fetcher.py  parser.py  normalize.py  pipeline.py  scrapy_spider.py  run_scrapy.py  fixture_site.py  store.py
  sample_data.py       synthetic offline sample
scripts/               init_db  ingest_all  fetch_datasets  build_lexicon  run_crawler  build_fixture_site  serve_fixture_site  verify_phase1  export_fsa_diagrams
tests/                 pytest suite
docs/                  DATASETS.md  DESIGN_NOTES.md  FINITE_AUTOMATA.md  phase1_report.md  schema.sql  er_diagram.mmd  fsa/
```

## Things to know before relying on the data
* **Not every source has full text.** Naukri and LinkedIn (until the job-summary join is done) have skill tags only; MyCareersFuture text is pre-lower-cased with stop-words removed. Each job carries `text_quality` (full / preprocessed / skills_only / none). Passage retrieval and sequence models in later phases use only the `full` sources.
* **Experience for LinkedIn/EMSCAD may be an approximate band** derived from seniority (`experience_source = 'seniority_map'`). Real values are `dataset` or `extracted`.
* **`cluster` is a title-keyword heuristic, not a label.** The only real classification labels are EMSCAD `function` (64% of rows) and MyCareersFuture `category`.
* **Synthetic data is always labelled** (`crawler_sources.source_type = 'synthetic'`) and must never be used for the evaluation chapters.
* Dates: MyCareersFuture is almost entirely May-June 2019 and LinkedIn spans 6 days, so "trending skills" can only compare two periods.

## Phase 2 - the IR engine (search, ranking, passages, PageRank)

```powershell
python scripts/compute_pagerank.py        # link graph + PageRank -> job_pagerank, job_links      (~1 min)
python scripts/build_index.py --write-db  # retrieval index (data/index/) + job_passages/job_terms  (~1 min)
python scripts/search_cli.py "Entry level machine learning jobs in Pune"
python scripts/search_cli.py "python machine learning sql" --compare        # LM vs DFR vs baseline vs hybrid, side by side
python scripts/search_cli.py "data analyst" --city Pune --min-exp 0 --max-exp 2 --model dfr
python scripts/verify_phase2.py           # integrity + measured latency/agreement -> docs/phase2_report.md
```
Models: `baseline`, `lm` (unigram, Dirichlet/JM), `lm_bigram`, `dfr` (PL2/InL2), `passage`, `hybrid`. Filters: city, country, role, skills, experience range, salary, employment type, sources, text quality. Re-run `build_index.py` after every re-ingest.

## Phase 3 - parallel / distributed retrieval, cluster routing, MapReduce

```powershell
python scripts/recompute_clusters.py      # once, after upgrading: applies the refined 12-cluster rules to the database
python scripts/build_index.py             # rebuild the index with the new clusters
python scripts/benchmark_parallel.py      # sequential vs parallel retrieval ON YOUR CORES (speedup, efficiency)  -> docs/parallel_benchmark.md
python scripts/run_mapreduce.py --workers 4          # skill-market analysis via MapReduce, validated against SQL -> data/analytics/market.json
python scripts/search_cli.py "python machine learning sql" --parts 4 --executor processes      # search 4 partitions in parallel
python scripts/search_cli.py "nurse" --route                                                    # cluster-based retrieval (experimental)
python scripts/verify_phase3.py           # exactness + routing + MapReduce + benchmark -> docs/phase3_report.md
```
Details, numbers and limits: `docs/PARALLEL_DISTRIBUTED.md`, `docs/ARCHITECTURE.md`. On Windows, scripts that start processes need the `if __name__ == "__main__":` guard (already present).

## Phase 4 - IR evaluation (needs YOUR relevance labels)

```powershell
python scripts/migrate_phase4.py                              # once, if your database was created before Phase 4
python scripts/build_index.py                                 # rebuild (the index now stores stable job identifiers)
python scripts/make_judgment_sheet.py                         # -> data/eval/judgments_to_label.xlsx  (477 jobs to label 0/1/2)
python scripts/evaluate_ir.py --judgments labeled.xlsx --run-label manual-1     # -> docs/phase4_report.md + ir_experiments table
python scripts/selftest_evaluation.py                         # the harness on RANDOM labels: shows it does not invent differences
```
Full workflow, statistics and the self-test: `docs/EVALUATION.md`.

## Phase 5 - deep learning (TensorFlow/Keras) - see `docs/DEEP_LEARNING.md` and `docs/phase5_report.md`

```powershell
pip install -r requirements.txt                                   # includes tensorflow-cpu, scikit-learn, matplotlib
python scripts/train_dl.py --prepare                              # leakage-free dataset (employer split, duplicates collapsed) + tokenizer
python scripts/train_dl.py --model baseline rnn birnn lstm gru --seeds 42 43 --skip-existing --budget 250     # resumable; run again until nothing is left
python scripts/run_gradient_experiment.py                         # vanishing / exploding gradients, clipping (measured, ~4 min)
python scripts/trace_gates.py                                     # LSTM/GRU gates on a real description (NumPy cells, checked against Keras)
python scripts/run_gate_demo.py --only short medium --seeds 0 1 2 # controlled SYNTHETIC memory task that makes the gates visible
python scripts/train_autoencoder.py                               # basic / deep / DENOISING autoencoders on skill vectors + regularisation ablation (~15 min)
python scripts/run_case_studies.py                                # conv-AE vs PCA, VAE, network-traffic anomaly (SYNTHETIC traffic)
python scripts/build_dl_embeddings.py                             # job embeddings for Phase 6 (text space + skill space)
python scripts/report_dl.py                                       # docs/phase5_report.md + charts + rows in model_experiments
```
What it delivers: RNN / BiRNN / LSTM / GRU on one skeleton for **job-function classification** (title excluded, duplicates collapsed, split by employer) with the syllabus comparison table
(accuracy, precision, recall, F1, training time, validation loss) and a bag-of-words reference; the gradient experiment; from-scratch NumPy LSTM/GRU verified against Keras;
denoising / deep / basic autoencoders with regularisation; convolutional-AE and VAE comparison, image compression, a separate network-traffic anomaly case study;
`DLEncoder` (embeddings + category prediction, nothing retrained at search time) and the embedding builder.
**Honest results to know about:** the recurrent models score below the bag-of-words baseline on this data; the denoising autoencoder ties (does not beat) a co-occurrence baseline;
the memory task and the network traffic are synthetic and labelled so. Details and numbers: `docs/phase5_report.md`.
NER and the resume analyzer (document/image processing, OCR) are in Phase 6.

## Where the project stands (updated 2026-10-13)
* **Phase 6 (recommenders, resume reading, skill gap)** is partly built (`jobintel/rec/`, `docs/PHASE6.md`). The user's own partial Phase 6 files have been reviewed and a merge rehearsed, **but not applied**; it waits for the user's approval (`docs/HANDOFF.md`, section 12).
* **Phase 7 (web application)**: Stages 1-2 of 5 are built - Home, Find Jobs, Job Details, accounts, Saved Jobs, Profile, Account settings, workspace, About/Privacy/Terms; Recommendations, Resume and Career Insights are labelled placeholders. See `docs/PHASE7.md`. To run it on your own computer: `pip install -r requirements.txt`, build the index (`python scripts/build_index.py --write-db`), set `SECRET_KEY`, then `python scripts/run_web.py` and open http://127.0.0.1:5000 (local use only; not a production server).
* Tests: `python -m pytest` (372 passed on 2026-10-13). Phase 8 (report, diagrams, testing document) is not started.
* `_backups/` holds the files replaced during Phase 7 and `_session_archive/` the rehearsal and scratch work that is not part of the running application.

## Next phases
Phase 7 Stages 3-5 (resume analyzer and recommendations, career insights, hardening) - then Phase 8 report.
