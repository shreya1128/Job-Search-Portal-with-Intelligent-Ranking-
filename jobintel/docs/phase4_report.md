# Phase 4 - IR evaluation report (manual-1)

*Generated 2026-10-14 02:43.  Relevance source: **manual**.  Collection: 3,951 jobs.  Queries evaluated: 13.  Binary relevance = grade >= 1.  Pool depth: 10.*

## How to read this report

* Every number is computed from the systems' ACTUAL ranked results and the relevance labels in the judgment sheet - nothing is typed in.
* Labels: 35 judged jobs (grade 2: 17, grade 1: 15, grade 0: 3); 0 pooled jobs unlabeled.
* Brackets in sections 1-2 are 95% bootstrap intervals over the queries; with about 20 queries they are WIDE, and bootstrap intervals are themselves a little optimistic at this size. Whether two systems differ is decided only in section 3.
* Recall is measured against the relevant jobs found in the judged pool, not against all relevant jobs in the collection (unknown).
* Jobs no pooled system returned were never judged and count as not relevant. `judged@10` shows how much of each system's top 10 carries a judgment; a system with a low value is penalised by the method, not necessarily by its quality.

## 1. The comparison asked for: baseline vs language model vs DFR vs hybrid IR

| system | P@5 | P@10 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| **keyword_baseline** | 0.292 [0.185, 0.415] | 0.192 [0.108, 0.292] | 0.717 [0.508, 0.897] | 0.800 [0.607, 0.958] | 0.729 [0.521, 0.917] | 0.655 [0.476, 0.816] |
| **lm_dirichlet** | 0.385 [0.200, 0.585] | 0.223 [0.115, 0.346] | 0.737 [0.512, 0.926] | 0.858 [0.667, 1.000] | 0.806 [0.583, 1.000] | 0.764 [0.581, 0.910] |
| **dfr_pl2** | 0.277 [0.123, 0.462] | 0.169 [0.085, 0.262] | 0.484 [0.262, 0.706] | 0.667 [0.426, 0.883] | 0.608 [0.350, 0.850] | 0.551 [0.332, 0.759] |
| **hybrid** | 0.385 [0.230, 0.554] | 0.208 [0.123, 0.300] | 0.809 [0.620, 0.956] | 0.904 [0.808, 0.983] | 0.816 [0.615, 1.000] | 0.775 [0.639, 0.895] |

> **(!) Unreliable systems:** dfr_inl2 (only 18% of its top-10 judged), dfr_pl2 (only 18% of its top-10 judged), hybrid (only 23% of its top-10 judged), hybrid_equal_weights (only 22% of its top-10 judged), hybrid_no_pagerank (only 23% of its top-10 judged), hybrid_text_only (only 24% of its top-10 judged), keyword_baseline (only 21% of its top-10 judged), lm_bigram (only 10% of its top-10 judged), lm_dirichlet (only 25% of its top-10 judged), lm_dirichlet_mu2000 (only 22% of its top-10 judged), lm_jm (only 15% of its top-10 judged), passage (only 18% of its top-10 judged). Their unjudged results count as NOT relevant, which lowers their scores for a reason that has nothing to do with quality (see the null self-test in docs/EVALUATION.md). No comparison involving them is reported as supported. Label the missing jobs with `scripts/make_judgment_sheet.py --extend <labeled sheet> --systems <name>` and evaluate again.


## 2. All systems and variants

| system | P@10 | MRR | nDCG@10 | judged@10 | median ms | queries |
|---|---|---|---|---|---|---|
| keyword_baseline (!) | 0.192 [0.108, 0.292] | 0.729 [0.521, 0.917] | 0.655 [0.476, 0.816] | 0.21 | 0.9 | 13 |
| lm_dirichlet (!) | 0.223 [0.115, 0.346] | 0.806 [0.583, 1.000] | 0.764 [0.581, 0.910] | 0.25 | 0.8 | 13 |
| lm_jm (!) | 0.138 [0.062, 0.223] | 0.490 [0.240, 0.750] | 0.428 [0.233, 0.631] | 0.15 | 1.0 | 13 |
| lm_dirichlet_mu2000 (!) | 0.192 [0.115, 0.277] | 0.701 [0.486, 0.896] | 0.676 [0.504, 0.828] | 0.22 | 1.1 | 13 |
| lm_bigram (!) | 0.085 [0.038, 0.138] | 0.442 [0.181, 0.694] | 0.344 [0.119, 0.585] | 0.10 | 6.1 | 13 |
| dfr_pl2 (!) | 0.169 [0.085, 0.262] | 0.608 [0.350, 0.850] | 0.551 [0.332, 0.759] | 0.18 | 0.6 | 13 |
| dfr_inl2 (!) | 0.169 [0.077, 0.269] | 0.538 [0.278, 0.788] | 0.485 [0.269, 0.702] | 0.18 | 0.6 | 13 |
| passage (!) | 0.162 [0.085, 0.246] | 0.493 [0.250, 0.743] | 0.467 [0.270, 0.662] | 0.18 | 69.9 | 13 |
| hybrid (!) | 0.208 [0.123, 0.300] | 0.816 [0.615, 1.000] | 0.775 [0.639, 0.895] | 0.23 | 35.4 | 13 |
| hybrid_no_pagerank (!) | 0.208 [0.123, 0.300] | 0.782 [0.562, 1.000] | 0.759 [0.595, 0.900] | 0.23 | 51.4 | 13 |
| hybrid_text_only (!) | 0.215 [0.115, 0.331] | 0.659 [0.413, 0.885] | 0.667 [0.471, 0.849] | 0.24 | 38.4 | 13 |
| hybrid_equal_weights (!) | 0.200 [0.115, 0.292] | 0.760 [0.548, 0.944] | 0.704 [0.567, 0.835] | 0.22 | 34.7 | 13 |

## 3. Paired comparisons (difference = first minus second, same queries)

A difference is called supported only if the sign-flip randomization test stays below 0.05 AFTER Holm correction over the 0 valid comparisons per metric; the bootstrap interval only describes how large the difference may be. With this many comparisons some nominal p < 0.05 are expected by chance alone.

| comparison | metric | result | wins / ties / losses |
|---|---|---|---|
| language model vs baseline | P@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 3 / 8 / 2 |
| language model vs baseline | MRR | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 4 / 6 / 2 |
| language model vs baseline | nDCG@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 7 / 3 / 2 |
| DFR vs baseline | P@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 4 / 4 / 5 |
| DFR vs baseline | MRR | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 5 / 3 / 4 |
| DFR vs baseline | nDCG@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 6 / 1 / 5 |
| hybrid vs baseline | P@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 4 / 6 / 3 |
| hybrid vs baseline | MRR | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 5 / 4 / 3 |
| hybrid vs baseline | nDCG@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 7 / 1 / 4 |
| hybrid vs language model | P@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 1 / 10 / 2 |
| hybrid vs language model | MRR | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 1 / 8 / 3 |
| hybrid vs language model | nDCG@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 3 / 2 / 7 |
| hybrid vs DFR | P@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 5 / 8 / 0 |
| hybrid vs DFR | MRR | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 5 / 7 / 0 |
| hybrid vs DFR | nDCG@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 9 / 2 / 1 |
| language model vs DFR | P@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 6 / 6 / 1 |
| language model vs DFR | MRR | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 4 / 7 / 1 |
| language model vs DFR | nDCG@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 8 / 3 / 1 |
| bigram vs unigram language model | P@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 0 / 6 / 7 |
| bigram vs unigram language model | MRR | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 0 / 6 / 6 |
| bigram vs unigram language model | nDCG@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 0 / 4 / 8 |
| PageRank effect (hybrid vs hybrid without PageRank) | P@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 0 / 13 / 0 |
| PageRank effect (hybrid vs hybrid without PageRank) | MRR | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 2 / 10 / 0 |
| PageRank effect (hybrid vs hybrid without PageRank) | nDCG@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 2 / 8 / 2 |
| skill/location/experience/PageRank signals (hybrid vs text-only hybrid) | P@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 1 / 11 / 1 |
| skill/location/experience/PageRank signals (hybrid vs text-only hybrid) | MRR | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 4 / 8 / 0 |
| skill/location/experience/PageRank signals (hybrid vs text-only hybrid) | nDCG@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 5 / 4 / 3 |
| default weights vs equal weights | P@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 1 / 12 / 0 |
| default weights vs equal weights | MRR | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 2 / 8 / 2 |
| default weights vs equal weights | nDCG@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 6 / 4 / 2 |
| Jelinek-Mercer vs Dirichlet smoothing | P@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 0 / 6 / 7 |
| Jelinek-Mercer vs Dirichlet smoothing | MRR | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 0 / 6 / 6 |
| Jelinek-Mercer vs Dirichlet smoothing | nDCG@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 0 / 2 / 10 |
| InL2 vs PL2 | P@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 1 / 11 / 1 |
| InL2 vs PL2 | MRR | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 0 / 9 / 3 |
| InL2 vs PL2 | nDCG@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 1 / 6 / 5 |
| Dirichlet mu=2000 vs mu=200 | P@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 0 / 11 / 2 |
| Dirichlet mu=2000 vs mu=200 | MRR | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 2 / 7 / 3 |
| Dirichlet mu=2000 vs mu=200 | nDCG@10 | UNRELIABLE - a compared system has unjudged results; no conclusion drawn | 2 / 4 / 6 |

## 4. Per query (nDCG@10 / P@10)

| query | relevant judged | keyword_baseline | lm_dirichlet | dfr_pl2 | hybrid |
|---|---|---|---|---|---|
| python machine learning sql | 7 | 0.25 / 0.3 | 0.99 / 0.7 | 0.75 / 0.4 | 0.85 / 0.5 |
| Entry level machine learning jobs in Pune | 1 | 0.00 / 0.0 | 0.00 / 0.0 | 0.29 / 0.1 | 1.00 / 0.1 |
| data analyst sql excel | 2 | 0.80 / 0.2 | 0.83 / 0.1 | 0.00 / 0.0 | 0.41 / 0.1 |
| senior data scientist tensorflow deep learning | 6 | 0.66 / 0.6 | 0.78 / 0.6 | 0.73 / 0.5 | 0.79 / 0.5 |
| customer service representative | 1 | 1.00 / 0.1 | 1.00 / 0.1 | 1.00 / 0.1 | 1.00 / 0.1 |
| financial analyst forecasting excel | 2 | 0.80 / 0.2 | 0.80 / 0.2 | 0.40 / 0.2 | 0.69 / 0.2 |
| python developer 3 years experience | 2 | 0.39 / 0.1 | 0.92 / 0.2 | 0.82 / 0.2 | 0.88 / 0.2 |
| business intelligence power bi | 5 | 0.80 / 0.5 | 0.87 / 0.4 | 0.62 / 0.3 | 0.89 / 0.4 |
| project manager agile scrum | 1 | 0.63 / 0.1 | 1.00 / 0.1 | 1.00 / 0.1 | 1.00 / 0.1 |
| primary school teacher | 0 | n/a | n/a | n/a | n/a |
| warehouse forklift operator | 3 | 0.53 / 0.2 | 1.00 / 0.3 | 1.00 / 0.3 | 0.97 / 0.3 |
| digital marketing manager | 1 | 1.00 / 0.1 | 0.36 / 0.1 | 0.00 / 0.0 | 0.32 / 0.1 |
| big data hadoop spark developer | 1 | 1.00 / 0.1 | 0.63 / 0.1 | 0.00 / 0.0 | 0.50 / 0.1 |

## 5. Limitations

* Small query set and ONE labeller (no inter-annotator agreement was measured).
* Pooling bias: only the top results of the pooled systems were judged; recall and nDCG are relative to that pool.
* The systems were not tuned on these queries (default parameters), which is what lets the comparison be read as a comparison; any later tuning must use different queries.
* The hybrid's skill / location / experience signals use information that the pure text models ignore - the ablations in section 3 separate those effects from the text models themselves.
* Queries without any relevant judged job are excluded from recall, MRR and nDCG.