# HANDOFF - continue this project in a new session

*Written at the end of a long session (sections 0-11 on 2026-10-10 after Phase 5; **section 12 added 2026-10-13 after Phase 7 Stage 1 - read it first, it supersedes the status table where they differ**). Read this file completely before touching anything. The previous conversation is NOT available to you; everything you need is here and in the zip `jobintel_project_handoff.zip`.*

## 0. First 10 minutes in the new session
1. Ask the user to upload: this file, `jobintel_project_handoff.zip`, and (only if you will need them) the datasets listed in section 6. Unzip to `/home/claude/jobintel`.
2. `cd /home/claude/jobintel && pip install --break-system-packages -r requirements.txt` (adds tensorflow-cpu, scikit-learn, matplotlib).
3. Run the fast tests: `python3 -m pytest -q` -> expect **319 passed** as of 2026-10-13 (170 at the end of Phase 5; the rest are the Phase 6 `rec` tests and the 98 Phase 7 web tests; the slow-marked test is deselected by `pytest.ini`; about 3-5 minutes because TensorFlow imports). `python3 -m pytest -m slow` runs the Scrapy-vs-BeautifulSoup test (1 passed).
4. If the task needs the database/index (Phases 5-7 do), rebuild the environment as in section 5.
5. Read `docs/DESIGN_NOTES.md` (every bug found so far and the rule it produced) before writing code.

## 1. The project in one paragraph
MCA academic project for TWO subjects, **Information Retrieval** and **Deep Learning**: "Job Market Skill Analysis and Job Recommendation System" - a small, professional job-search + job-market-intelligence + recommendation platform (NOT a CRUD app). Pipeline: JOB DATA -> WEB CRAWLING -> PREPROCESSING -> INFORMATION EXTRACTION -> INFORMATION RETRIEVAL -> DEEP LEARNING -> RECOMMENDATION -> SKILL GAP -> JOB MARKET INTELLIGENCE -> WEB APPLICATION. Target balance: IR 45% / DL 35% / application 20%. Marks depend on every syllabus topic being meaningfully demonstrated, so topics must not be dropped or silently changed.

## 2. Requirements digest (condensed from the user's four specification documents)
**IR** - finite automata (patterns: experience, degrees, roles, skills, salary, locations) and language models: unigram + bigram (trigram optional), multinomial distributions, query likelihood, smoothing (Jelinek-Mercer/Dirichlet), **DFR**, passage retrieval with highlighting ("WHY THIS JOB MATCHED"), ranking, keyword baseline, **PageRank** over Company -> Career page -> Job -> Related job, parallel query processing over partitions (sequential vs parallel: time, speedup, efficiency), MapReduce-style skill analysis, search-engine architecture, cluster-based architecture, distributed-architecture simulation, crawler structure with **Scrapy and BeautifulSoup**, ranking evaluation (P@5, P@10, R@5, R@10, MRR, nDCG, retrieval time; baseline vs LM vs DFR vs hybrid; PageRank impact; MANUAL relevance labels, proxy labels reported separately), search-result explanation, filters (location, experience, role, salary, employment type, skills), `/ir-experiments` dashboard, IR+DL integration, recommender integration, configurable final IR score (weights NOT claimed optimal), 20-chapter IR documentation (objective, input, processing, output, math, implementation, result, limitations for every algorithm).
**DL** - sequence learning, unfolding, RNN, **BiRNN**, vanishing/exploding gradients (experiment), **LSTM** (input/forget/output gates, selective write/forget/read, cell state), **GRU**, comparison table of RNN/BiRNN/LSTM/GRU (accuracy, precision, recall, F1, training time, validation loss) on a REAL task (job-description classification), **autoencoders** (basic, deep, **denoising** = main, regularization; small comparison/explanation of convolutional and variational AE; image-compression application; network-traffic anomaly case study kept separate from the job workflow), project pipeline, **NER**, resume document/image processing (region/section detection, OCR), recommender systems, brief case-study text for image captioning / facial recognition / self-driving (NOT application modules). Use TensorFlow/Keras; do NOT use BERT/Transformers as the primary DL solution; do NOT use TF-IDF/BM25 as the primary retrieval.
**Recommendation / KB** - content-based, collaborative (viewed/clicked/saved/applied; synthetic interactions clearly labelled), knowledge-based (skill/role knowledge graph, cold start), hybrid of 11 normalised, configurable signals (IR, LM, DFR, passage, PageRank, DL similarity, content, collaborative, knowledge, skill match, user preference); evaluate (P@K, R@K, coverage, hit rate). **Skill-gap analysis**: matching vs missing skills, importance, frequency of the missing skill in similar jobs, related skills from the knowledge base. **Job-market analytics**: top skills, by role/location/experience, co-occurrence, distributions, trends over time if dates exist.
**Application** - Python, **Flask**, **MySQL**, HTML/CSS/JS/Bootstrap, Pandas, NumPy, spaCy/NLTK, TensorFlow/Keras, Scrapy, BeautifulSoup, scikit-learn, Plotly/Matplotlib. Ten pages: Landing (headline "Understand the Job Market. Find Jobs That Fit You.", buttons "Explore Jobs" and "Analyze Resume"), User Dashboard, Job Search, Job Details, Recommendations, Market Intelligence, Skill Gap Analysis, Resume Analyzer, IR Experiments, DL Experiments. Loading/empty states, input validation, error handling, secrets in environment variables.
**Database** - 20 normalised MySQL tables (implemented as 21: + `occupations`).
**Rules** - never fabricate or hard-code results/recommendations/scores; never retrain per search; save models and preprocessing objects; offline sample dataset; label synthetic data; modular code; live scraping never the only source; one coherent system; build and test incrementally.
**Deliverables** - README, install guide, requirements, dataset docs, DB schema + ER diagram, architecture + data-flow diagrams, IR/DL algorithm explanations with maths, training docs, experiment results, screenshots, user workflow, testing docs, limitations, future scope, syllabus-to-feature map.

## 3. Status
| Phase | Content | Status |
|---|---|---|
| 1 | schema (21 tables), ingestion of 4 real job datasets, preprocessing, finite-automata extractor, knowledge base (curated + ESCO + O*NET), Scrapy + BeautifulSoup crawlers, fixture site, synthetic sample | **done, tested** |
| 2 | index, unigram/bigram LM, DFR, passages + highlighting, query processor (validation DFA), PageRank, hybrid ranker, explained results, CLI | **done, tested** |
| 3 | partitions, parallel/distributed engine (sequential/threads/processes), cluster router, MapReduce (validated against SQL), benchmark | **done, tested** (speedup NOT measurable in the sandbox: 1 core) |
| 4 | metrics, pooled judgment sheet, evaluation runner, significance tests, report | **built and tested; numbers pending the user's relevance labels** |
| 5 | deep learning (RNN/BiRNN/LSTM/GRU comparison, gradients, NumPy cells, autoencoders, case studies, encoder + embeddings, report) | **done, tested** - see section 4 |
| 6 | recommenders, NER, resume analyzer, skill gap, synthetic interactions, DL-similarity integration | **partial**: `jobintel/rec/` exists and is tested (see `docs/PHASE6.md`); the user's own partial Phase 6 upload has been reviewed and a merge REHEARSED but **NOT approved and NOT applied** (section 12) |
| 7 | Flask application (redefined by the user's Master Product Direction: 7 navigation items, not 10 pages) | **Stage 1 of 5 built and verified** (section 12, `docs/PHASE7.md`) |
| 8 | report, diagrams, syllabus map, screenshots, testing doc | not started |

## 4. Phase 5 - deep learning: DONE (see `docs/DEEP_LEARNING.md` and `docs/phase5_report.md`, both generated/verified from the saved results)
**Task (never change it without being asked):** classify a job's FUNCTION from its description with RNN / BiRNN / LSTM / GRU; EMSCAD labels, title excluded, near-duplicates collapsed, split by employer (5,802 jobs, 7 classes, train 4060 / val 810 / test 932).
**Delivered code** (`jobintel/dl/`): `data, tokenizer, models, train, baseline, registry` (earlier) + `gradients` (vanishing/exploding experiment with tf.GradientTape, clipping), `numpy_rnn` (from-scratch SimpleRNN/BiRNN/LSTM/GRU equal to Keras < 1e-5), `gate_demo` (SYNTHETIC controlled memory task), `autoencoder` (basic / deep / denoising + regularisation ablation on 1,500-skill multi-hot vectors, recovery@k vs baselines, paired employer-clustered bootstrap, k-NN purity, `suggest_skills`), `case_studies` (conv-AE vs PCA, VAE, SYNTHETIC network-traffic anomaly, `load_cic_ids` for real CSVs), `encoder` (`DLEncoder`: best-by-validation model, `embed`, `predict_category`).
**Scripts:** `train_dl.py` (resumable; `--skip-existing` now also requires the saved model file), `run_gradient_experiment.py`, `trace_gates.py`, `run_gate_demo.py`, `train_autoencoder.py` (saves progress after every model, `--eval-only`), `run_case_studies.py`, `build_dl_embeddings.py` (text space + skill space + sanity check), `report_dl.py` (report, charts, rows in `model_experiments`).
**Measured results** (test set; mean (min-max) over 2 seeds):

| model | accuracy | macro-F1 | val loss |
|---|---|---|---|
| rnn | 0.318 (0.314-0.321) | 0.273 (0.269-0.276) | 1.549 (1.532-1.566) |
| birnn | 0.416 (0.364-0.468) | 0.363 (0.307-0.420) | 1.376 (1.263-1.489) |
| lstm | 0.545 (0.521-0.569) | 0.509 (0.497-0.520) | 1.089 (1.072-1.105) |
| gru | 0.495 (0.452-0.539) | 0.459 (0.394-0.524) | 1.305 (1.240-1.370) |
| bag-of-words logistic regression (reference) | 0.628 | 0.645 | 0.754 |

* All recurrent models are BELOW the bag-of-words baseline; LSTM and GRU ranges overlap; seed variation is large (report this honestly, do not tune the baseline down). Deployed model = lowest validation loss = `lstm_s43`.
* Gradients: log10(first/last-step gradient) at T=100, scale 1: SimpleRNN tanh -17.2, LSTM -8.8, GRU -21.1 (default GRU is the WORST), LSTM with forget bias 5 +0.3, GRU with update bias 5 -0.3; ReLU RNN with scale 2 explodes (+18.0). In a short real-data training run clipping did not change whether training diverged (no run diverged).
* Gates on real text barely move (all four gate means stay within about 0.1-0.15 of their initial values); on the SYNTHETIC memory task all models solve T=30, SimpleRNN and LSTM solve T=50 while the GRU stays at chance (consistent with the gradient table), nobody solves T=80 in 25 epochs.
* Autoencoders (test recall@10 of dropped skills): random 0.001, popularity 0.182, co-occurrence 0.303, denoising AE 0.307 (paired difference to co-occurrence +0.0040, 95% interval [-0.0024, +0.0106] -> a TIE, not a win); deep AE without L2/dropout collapses (0.138); spurious-skill AUC denoising 0.922 vs co-occurrence 0.907.
* Case studies: conv-AE beats PCA at k=4/8/16; VAE samples all 10 digits; synthetic traffic AE AUC 0.966 (PCA 0.983 is slightly better); results describe the generator only.
* Embeddings: text embedding 64-d from `lstm_s43`, 10-NN same-function purity 0.567 (chance 0.178) on held-out employers. `data/dl/embeddings_text.npz` (17,800 jobs) and `embeddings_skills.npz` (42,589 jobs) are keyed by (source, external_id).
**Deviations to remember:** MyCareersFuture could not be downloaded in the last session (GitHub 403 from the sandbox egress policy), so (a) category purity of the autoencoder uses EMSCAD labels, (b) the database/index in that session had no MCF (42,075 indexed documents instead of 60,808); everything is reproducible with the MCF file present. AE per-epoch loss curves were not retained (only epochs / best val loss / seconds from the log). The SimpleRNN `.keras` weight files (`rnn_s42`, `rnn_s43`) are not on disk - only their result JSONs; the other six sequence models and all five autoencoders are saved. `DLEncoder` is unaffected (best-by-validation is `lstm_s43`); to regenerate, run `train_dl.py --model rnn --seeds 42 43` (this overwrites the two result JSONs, so the report must be regenerated).
**Phase 6 starting points:** use `DLEncoder.embed` / `embeddings_text.npz` for the DL-similarity signal (fall back to `embeddings_skills.npz`, never mix spaces), `autoencoder.suggest_skills` for skill-gap suggestions, `DLEncoder.predict_category` for filters / cluster routing; NER (spaCy EntityRuler from the automata/lexicon, weakly supervised), resume analyzer (PyMuPDF/pdfplumber for text PDFs; OpenCV region detection + Tesseract OCR for scans), recommenders (content / collaborative with SYNTHETIC labelled interactions / knowledge-based), 11-signal hybrid, skill gap.

## 5. Environment (the sandbox of the previous session) - rebuild recipe
Constraints: **300 s limit per tool call**; 1 CPU core; 4 GB RAM; `/bin/sh` is dash (no `{a,b}` brace expansion); background processes die between calls; pip needs `--break-system-packages`; network allows PyPI/GitHub only (no Kaggle / Hugging Face / onetcenter.org / esco site).
MySQL-compatible test server = **MariaDB 10.11** (`apt-get install -y mariadb-server`, runs as root). First-time setup:
```
mkdir -p /tmp/mysqldata /tmp/mysqlrun /run/mysqld && chown -R mysql:mysql /tmp/mysqldata /tmp/mysqlrun /run/mysqld
mariadb-install-db --user=mysql --datadir=/tmp/mysqldata --auth-root-authentication-method=normal
setsid nohup mariadbd --user=mysql --datadir=/tmp/mysqldata --socket=/tmp/mysqlrun/mysql.sock --pid-file=/tmp/mysqlrun/mysqld.pid --port=3307 --bind-address=127.0.0.1 --innodb-use-native-aio=0 >/tmp/mdb.log 2>&1 </dev/null &
mariadb -uroot -S/tmp/mysqlrun/mysql.sock -e "CREATE DATABASE jobintel CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci; CREATE USER 'jobintel'@'127.0.0.1' IDENTIFIED BY 'testpw'; CREATE USER 'jobintel'@'localhost' IDENTIFIED BY 'testpw'; GRANT ALL ON jobintel.* TO 'jobintel'@'127.0.0.1'; GRANT ALL ON jobintel.* TO 'jobintel'@'localhost';"
```
(`--innodb-use-native-aio=0` and the `/run/mysqld` directory were both needed.) `tools/sandbox/startdb.sh` restarts the server idempotently - **call it at the top of every tool call that touches the database**; `tools/sandbox/env.sh` holds the environment variables (DB on port 3307, dataset paths) - edit the paths to wherever the files are, then `. tools/sandbox/env.sh`.
Rebuild the data (split into several calls because of the 300 s limit):
```
python3 scripts/init_db.py --drop
python3 scripts/ingest_all.py --rebuild-lexicon --only knowledge naukri linkedin resumes     # ~1 min
python3 scripts/ingest_all.py --only emscad                                                  # ~100 s
python3 scripts/ingest_all.py --only mcf gazetteer                                           # ~90 s
python3 scripts/compute_pagerank.py ; python3 scripts/recompute_clusters.py ; python3 scripts/build_index.py --write-db
python3 scripts/verify_phase1.py ; python3 scripts/verify_phase2.py ; python3 scripts/verify_phase3.py     # each should end ALL CHECKS PASSED
python3 scripts/train_dl.py --prepare                                                        # rebuilds data/dl dataset + tokenizer
```
**Sandbox facts learned in the last session:** the container can be restored from a snapshot between turns - files survive but ALL background processes (incl. MariaDB) die: re-run `tools/sandbox/startdb.sh`, and run long jobs as resumable chains (`setsid nohup sh script.sh` with per-step marker files, check `ps` before assuming they run). Never `pkill -f` a pattern that appears in your own command line. `fetch_datasets.py --only mcf` returned HTTP 403 from the egress proxy (policy; do not route around it) - ask the user for the MCF file or use the copy in their environment.

## 6. Data
| dataset | how to get | notes |
|---|---|---|
| Naukri (12,000 jobs, skills only, India) | the user's file `naukri_data_science_jobs_india.csv` | no description text |
| EMSCAD / fake_job_postings (17,880, full text) | `scripts/fetch_datasets.py --only emscad` or the user's `archive.zip` | the DL task uses its `function` label |
| MyCareersFuture (20,298, Singapore, text pre-lowercased) | `fetch_datasets.py --only mcf` | job_id not unique; dates are May-June 2019 |
| LinkedIn sample (13,989, skills only) | `linkedin_sample_meta.csv` (given to the user; copy to `data/raw/linkedin/`) | full text needs the user to run `attach_summaries.py` against their huge `job_summary.csv` (not done) |
| ESCO CSVs + O*NET 31.0 Excel | `fetch_datasets.py --only esco`; O*NET = user's `db_31_0_excel.zip` | `P_3.rdf` is only the ESCO data model (OWL/SKOS) |
| Resumes (2,484, 24 categories, PDFs + CSV) | the user's `archive__2__1.zip` | text PDFs only; no scans |
Not available: CIC-IDS2017 (use synthetic traffic, labelled), real scanned resumes, real interaction logs (generate synthetic ones, labelled).

## 7. Decisions and bugs that must not be forgotten (details in `docs/DESIGN_NOTES.md`)
* **Duplicates** = same title AND employer AND place AND identical/near-identical text (never text alone). Flagged (`duplicate_of`), not deleted.
* **Placeholder employers** ("Company Undisclosed" ...) are "no company" (they created a fake giant employer and fake PageRank authority).
* **Scorers must restrict postings to the candidate set** (a Phase-2 bug added filtered-out documents' scores to the last candidate); a document's score must not depend on other candidates (test exists).
* **Top-k ties are deterministic** (`models.top_positions`: score desc, then document order); `np.argpartition` alone made partitioned and single-index search disagree.
* **Any script that starts processes needs `if __name__ == "__main__":`** (Windows `spawn` re-imports it).
* **Job ids are not portable between databases** - use (`source`, `external_id`); the judgment sheet and index store them.
* LinkedIn "experience" is an approximation from seniority (`experience_source = seniority_map`) - never show it as stated experience.
* Statistics: decisions use a paired sign-flip randomization test + Holm correction (the bootstrap interval was too optimistic with 20 queries); systems with unjudged top-10 results are flagged and no conclusion is drawn (measured: 30.6% false "significant" results under random labels).
* Synthetic data (`sample_synthetic`, `fixture_crawl`) is excluded from the index and must NEVER be used for evaluation.
* Verification scripts must not contain checks that cannot fail (one vacuous check was found and replaced).
* Cluster routing is opt-in: it misrouted `devops engineer docker kubernetes aws` (overlap 0.20); the Phase-5 classifier is the intended improvement.

## 8. Measured so far (reports in `docs/`)
Index 60,808 documents / 84,106 terms / 6.9 M tokens / 4.7 M postings; query latency median about 4 ms (baseline/LM/DFR), 16 ms (bigram), 34 ms (hybrid with DB passages). Parallel search returned identical results to single-process in 376 comparisons (6 models, filters, routing, real worker processes). MapReduce equals SQL (12,414 skill counts, spot checks) and rebuilds the inverted index exactly. Cluster routing: 18% of documents searched, top-10 overlap 0.88 on average. Null self-test of the evaluation harness: 2.9% / 0.7% false positives (raw / Holm) for pooled systems. **Phase 5 measured** (section 4). **Not measured**: IR relevance metrics (needs labels), real multi-core speedup (benchmark script exists), extraction precision/recall, anything on Oracle MySQL 8 (only MariaDB was used), recurrent models with more than 2 seeds, anything on real network traffic.

## 9. Working agreements with this user
* The user is on **Windows** (paths like `C:\Users\shrey\Downloads`); give PowerShell commands, quote paths with spaces, `.env` paths with forward slashes.
* "Continue" means: do the next piece of work and deliver the files; do NOT produce long audits/status reports unless asked (they asked for one once and then said not to).
* Each phase is delivered as a zip (present it with `present_files`). Keep summaries short, state limitations honestly, ask at most one question.
* The user wants every original requirement kept (marks depend on it): never swap the DL task, drop a topic or change the scope without being asked. They previously corrected an unrequested change.
* No fabricated metrics; synthetic data labelled; honest reporting of negative results (e.g. recurrent models below the baseline).

## 10. Blocked on / waiting for the user
1. **Relevance labels**: `relevance_judgments_to_label.xlsx` (477 jobs, 20 queries, 0/1/2). Then `scripts/evaluate_ir.py --judgments labeled.xlsx` (and `make_judgment_sheet.py --extend` on their machine). They were also offered a clearly-marked "assistant-judged" first pass - undecided.
2. They have never confirmed running Phases 1-3 on their own machine (instructions were given; MySQL 8 unverified).
3. Optional: scanned resumes (for OCR/region detection), CIC-IDS2017 flows, running `scripts/benchmark_parallel.py` on a multi-core machine, joining LinkedIn job summaries.

## 11. File map
`jobintel/` db, processing (text, FSA engine, patterns, dedupe), extraction (lexicon, extractor), ingest (loaders, load_db, knowledge, curated_kb, resumes, gazetteer), crawler (frontier, fetcher, parser, pipeline, scrapy_spider, fixture_site, store), ir (index, models, passages, query, pagerank, ranker, engine, partition, distributed, cluster, benchmark), mapreduce, eval, dl (done), sample_data.py - `scripts/` (init_db, ingest_all, fetch_datasets, build_lexicon, run_crawler, compute_pagerank, recompute_clusters, build_index, search_cli, run_mapreduce, benchmark_parallel, make_judgment_sheet, evaluate_ir, selftest_evaluation, train_dl, verify_phase1/2/3, ...) - `tests/` (134 fast + 1 slow) - `docs/` (DATASETS, DESIGN_NOTES, FINITE_AUTOMATA, IR_ENGINE, ARCHITECTURE, PARALLEL_DISTRIBUTED, EVALUATION, phase1-3 reports, parallel_benchmark, schema.sql, er_diagram.mmd, fsa/*.dot) - `tools/sandbox/` - `data/eval/queries.json`, `data/dl/{tokenizer.json,classification_info.json,results/*.json}`.
Also given to the user earlier (in their downloads): `sample_linkedin.py`, `attach_summaries.py`, `profile_job_summary.py`, `linkedin_sample_meta.csv`, `relevance_judgments_to_label.xlsx`.

## 12. State at 2026-10-13 (Phase 6 partial, Phase 7 Stages 1-2) - READ THIS FIRST

**The governing document is now the user's "MASTER PRODUCT DIRECTION" (2026-10-11), not the original ten-page list.** Ask the user to re-upload it if it is not in the conversation. Its rules, as understood: preserve the existing project and treat the newest files as the source of truth (never overwrite a newer file with an older snapshot; back up whatever is replaced); small reviewable stages; navigation exactly **Home | Find Jobs | Recommendations | Resume | Career Insights | Saved Jobs | Profile** (no IR/DL experiment pages in the navigation; the IR and DL algorithms stay inside the product, no comparison dashboards by default); near-black charcoal background, warm-white text, muted-grey secondary text, one restrained amber accent, sage for positives, thin borders, no purple gradients / neon / glow / sparkles / particles; accessible, keyboard-usable, responsive; Home must not duplicate other pages; **no fabricated statistics, testimonials, logos, match percentages or synthetic data presented as real**; never show stack traces or shell commands to users; do not claim authentication works until it is tested; do not delete existing tests, results or infrastructure without the user's approval.

### Phase 7 - what exists
Stage 1 (design system, Home, Find Jobs, Job Details, About/Privacy/Terms/status, login/register, placeholders for the rest) and **Stage 2 (Saved Jobs `/saved`, Profile `/profile`, Account settings `/account` with change password / download my data / delete account, leaner workspace `/dashboard`)** are built and verified (full suite 372 passed, 0 failed on SQLite; Stage 2 screenshots in `_session_archive/2026-10-12_phase7_session_scratch/screenshots_dark_stage2/`): **`docs/PHASE7.md` is the complete description** (pages and routes, design tokens, honesty decisions, code map, how to run with `scripts/run_web.py`, exactly what was tested, what was not). Screenshots are in the Phase 7 delivery zip and `_session_archive/2026-10-12_phase7_session_scratch/screenshots_dark_stage1/final_shots3/`. The user has asked to be shown each stage before the next starts: **Stage 1 and Stage 2 await their visual review.** Stages ahead: 3 Resume Analyzer + Recommendations, 4 Career Insights (+ skill gap), 5 hardening and packaging. Placeholders already show "Not built yet" with the stage number.
Earlier work this replaced: `_backups/web_light_theme_phase7_2026-10-11/` (the light-theme web package of the first Phase 7 attempt, delivered earlier as `jobintel_phase7_partial.zip`); its recommender page survives as `jobintel/web/views/recommend_draft.py` (not registered) and the experiments pages as `views/experiments.py` (not registered). Nothing was deleted.

### Phase 6 merge - NOT applied (the user's nine conditions are binding)
The user supplied a partial Phase 6 (`jobintel_phase6_additions.zip`, `PHASE6.md`, `PHASE5B_MCF.md`, relevance labels, Phase 5B v2 DL models). They approved *preparing* the merge but not applying it "exactly as written", and set nine conditions; **do not apply anything until they approve a revised plan**. The conditions (condensed): (1) keep the upload security checks and equivalent tests in the new pipeline before removing the old; (2) inspect dependencies before removing `UserProfile.document` / `ResumeProfile.raw_text`; (3) IR/DL stay internal, no comparison dashboards; (4) back up every replaced file, never overwrite newer with older; (5) verify recommender, v2 DL adapter, web backend and DB compatibility with real integration tests; (6) run the full suite plus resume upload, malformed and oversized files, PDF page limit, auth, access control, recommendation, skill-gap and search tests on a real database; (7) no synthetic interactions, fabricated recommendations or AI-judged labels shown as real; (8) MyCareersFuture stays explicitly unavailable until loaded and validated; (9) record unresolved licensing and deployment dependencies (including the PDF library) in the handoff and make no production-readiness claim.
The rehearsal (scratch copy only) is saved in `_session_archive/2026-10-12_phase6_merge_rehearsal_NOT_APPLIED/` with its README: 4 files replaced, 5 added in `rec/`, tests and scripts, a prototype `upload_guard.py` with 44 tests, 22 synthetic hostile inputs and logs; the rehearsed tree passed 276 of 277 tests (1 skipped: PyMuPDF not installed). One edit would touch the web backend (the DL loader `Backend.dl` switching to `DLSkillSimilarity.from_files(..., "v2")`); it must be re-applied by hand to the *current* `backend.py` (the saved patch is against the old one). The user's `read_pdf` uses PyMuPDF (AGPL); the proposed swap to pypdfium2 also awaits approval.
Condition 9 items (unresolved): see `docs/PHASE7.md` section 7, item 4.

### Facts about the sandbox that cost time
`tools/sandbox/startdb.sh` and `env.sh` start the MariaDB 10.11 test server (port 3307) and set the `DB_*` / data paths; the server dies between tool calls, so start it in the same call that needs it. 300 s limit per call (run the full suite in the background and poll). Chromium for screenshots is at `/opt/pw-browsers/chromium` (Playwright, `tools/phase7_scratch/shoot.py`). `JOBINTEL_TEST_DB_URL=mysql+pymysql://user:pw@host:port/some_test_db python -m pytest tests/test_web_account.py` runs the web tests on a real MySQL/MariaDB database (the name must contain "test"; every table in it is dropped). Jinja gotchas found: a template variable must not share a name with an imported macro (`errors`, `unavailable`, `empty` all caused silent bugs); imported macros cannot see the page context.

### Blocked on / waiting for the user (in addition to section 10)
1. Visual review of Phase 7 Stages 1 and 2 (then Stage 3 may start). Decision: add DB columns for certifications / several preferred locations / employment-type / job-area preferences? (`docs/PHASE7.md` 7.1a). Not built: e-mail change, e-mail password reset (no SMTP). Stage 3 must update `export_user_data`, the Account wording and `delete_account` together when recommendations / skill gaps / resume are stored (`docs/PHASE7.md` 7.1c). The all-web-tests-on-MariaDB re-run after the last template rename is not re-confirmed (Stage 2's 51 tests passed on MariaDB earlier).
2. Answer on the revised Phase 6 merge plan; approve or refuse the PyMuPDF -> pypdfium2 edit.
3. Review of the draft Privacy and Terms text (`templates/privacy.html`, `terms.html`).
4. Permission to clear this workspace's 10 anonymous test searches (not needed on their own machine).
