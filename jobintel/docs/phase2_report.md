# Phase 2 verification report

## Index

`{'documents': 60808, 'terms': 84106, 'tokens': 6915833, 'avg_doc_len': 113.7, 'postings': 4749803, 'has_pagerank': True}`

## Integrity checks

- PASS - index holds exactly the indexable jobs of the database (60,808 vs 60,808)
- PASS - sum of collection frequencies = total tokens
- PASS - df <= N and df <= cf for every term
- PASS - postings sorted by document inside every term
- PASS - PageRank stored for every document and positive
- PASS - no fraudulent / synthetic job in the index
- PASS - job_passages and job_terms tables are populated (84,381 passages, 2,113,958 term rows)
- PASS - every query returns finite-scored results under all 6 models ([])
- PASS - hybrid effective weights always sum to 1
- PASS - results never repeat a job within one ranking
- PASS - best passage contains at least one query term (full-text results) (33/33)
- PASS - every hybrid result carries reasons and scores
- PASS - city filter returns only jobs located in that city

## Latency (ms, 16 queries x 3 runs, 60,808 documents, single process)

| model | median | p95 | max |
|---|---|---|---|
| baseline | 3.8 | 8.3 | 14.6 |
| lm | 4.2 | 7.8 | 10.0 |
| lm_bigram | 16.3 | 50.0 | 73.0 |
| dfr | 4.3 | 7.2 | 13.0 |
| passage | 35.0 | 86.3 | 113.5 |
| hybrid | 33.9 | 77.8 | 100.7 |

## How much do the models disagree?  (mean overlap of the top-10 sets, 16 queries)

This shows the models rank DIFFERENTLY. It does **not** show which one is better - that is measured with relevance labels in Phase 4.

| | baseline | lm | lm_bigram | dfr | passage | hybrid |
|---|---|---|---|---|---|---|
| **baseline** | 1.00 | 0.36 | 0.15 | 0.09 | 0.18 | 0.20 |
| **lm** | 0.36 | 1.00 | 0.31 | 0.48 | 0.52 | 0.60 |
| **lm_bigram** | 0.15 | 0.31 | 1.00 | 0.27 | 0.24 | 0.28 |
| **dfr** | 0.09 | 0.48 | 0.27 | 1.00 | 0.49 | 0.59 |
| **passage** | 0.18 | 0.52 | 0.24 | 0.49 | 1.00 | 0.53 |
| **hybrid** | 0.20 | 0.60 | 0.28 | 0.59 | 0.53 | 1.00 |

## PageRank

- documents: 60,808; min 1.64e-06, median 4.37e-06, max 1.41e-03 (ratio max/median = 322x)
- correlation between log(number of postings) and log(company PageRank) over 4,207 employers: **0.65** - in this constructed graph PageRank mostly reflects posting volume/connectivity (prominent employers, recruiters, job boards), not quality; whether it helps retrieval is an experiment for Phase 4.

### Sample explained result (hybrid)

`Data Analyst - Machine Learning/ Python` / Finxera / Chandigarh - score 0.810

- signals: {lm: 0.79, dfr: 1.00, passage: 0.77, skill: 1.00, pagerank: 0.30}
- ✓ 'python' matched
- ✓ 'machine' matched
- ✓ 'learning' matched
- ✓ 'sql' matched
- ✓ relevant passage found
- ✓ skill: python
- ✓ skill: machine learning
- ✓ skill: sql
- passage: Skills: Business Intelligence, Data Visualization, Data Analyst, Data Warehousing, <mark>Machine Learning</mark>, <mark>Python</mark>, <mark>SQL</mark>

**Result: ALL CHECKS PASSED**