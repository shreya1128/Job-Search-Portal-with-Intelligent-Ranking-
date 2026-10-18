# Phase 3 verification report

Machine: 1 CPU core(s) - Linux-6.18.44-fc-v64-x86_64-with-glibc2.39 - Python 3.12.3 - NumPy 2.4.4

Index: `{'documents': 60808, 'terms': 84106, 'tokens': 6915833, 'avg_doc_len': 113.7, 'postings': 4749803, 'has_pagerank': True}`

## 1. Partitioned / parallel search equals single-process search

- PASS - 4 range partitions cover every document exactly once (sizes [15202, 15202, 15202, 15202])
- PASS - postings are partitioned without loss (4,749,803 postings)
- PASS - distributed ranking is IDENTICAL to the single-process ranking (job ids, scores, candidate counts) (376 comparisons over 4 configurations (sequential P=4, round-robin P=3, threads P=4, real processes P=2 started with 'spawn'), all 6 models, with filters and cluster routing; 15s)

## 2. Cluster-based retrieval (route the query to the most likely clusters, then search only those)

Cluster sizes: other 17,847, data_ai 11,190, software_it 9,793, sales_marketing 4,788, hr_admin 4,095, finance 4,077, healthcare 3,431, engineering_manufacturing 2,231, education 1,352, retail_hospitality 1,041, logistics_transport 744, legal_public 219

Clusters are title-keyword heuristics (the four named in the brief plus eight more), not ground truth.

| query | clusters searched (posterior of the best) | docs searched | candidates full -> routed | LM ms full / routed | top-10 overlap | full top-10 still reachable |
|---|---|---|---|---|---|---|
| python machine learning sql | data_ai (1.00) | 18% | 19,762 -> 7,463 | 3.9 / 3.9 | 1.00 | 1.00 |
| Entry level machine learning jobs in Pune | data_ai (0.95) | 18% | 11,116 -> 3,319 | 2.2 / 2.3 | 0.80 | 0.80 |
| data analyst sql excel | data_ai (1.00) | 18% | 37,724 -> 10,780 | 6.7 / 6.4 | 0.90 | 0.90 |
| registered nurse | healthcare (1.00) | 6% | 2,902 -> 1,910 | 0.9 / 1.0 | 0.90 | 0.90 |
| accountant tax | finance (0.99) | 7% | 9,422 -> 2,730 | 1.7 / 1.7 | 1.00 | 1.00 |
| senior data scientist tensorflow deep learni | data_ai (1.00) | 18% | 29,889 -> 10,980 | 4.9 / 4.9 | 1.00 | 1.00 |
| java spring boot microservices | software_it (0.91) | 16% | 5,090 -> 2,422 | 1.3 / 1.5 | 0.90 | 0.90 |
| devops engineer docker kubernetes aws | data_ai (0.91) | 18% | 19,583 -> 6,157 | 3.6 / 3.6 | 0.20 | 0.20 |
| customer service representative | sales_marketing, hr_admin, logistics_transport (0.31) | 16% | 25,486 -> 6,521 | 4.4 / 4.7 | 1.00 | 1.00 |
| financial analyst forecasting excel | finance (0.90) | 7% | 25,855 -> 3,114 | 4.3 / 4.0 | 1.00 | 1.00 |
| python developer 3 years experience | data_ai, software_it (0.71) | 35% | 29,203 -> 13,700 | 4.5 / 4.8 | 1.00 | 1.00 |
| business intelligence power bi | data_ai (1.00) | 18% | 22,425 -> 3,925 | 3.9 / 3.7 | 0.80 | 0.80 |
| nursing patient care icu | healthcare (1.00) | 6% | 5,831 -> 2,622 | 1.4 / 1.5 | 1.00 | 1.00 |
| react javascript frontend developer | software_it (0.99) | 16% | 25,344 -> 7,287 | 4.0 / 4.2 | 1.00 | 1.00 |
| cybersecurity analyst | data_ai, software_it (0.56) | 35% | 7,983 -> 3,619 | 1.4 / 1.7 | 0.90 | 0.90 |
| project manager agile scrum | data_ai, software_it, other (0.55) | 64% | 37,291 -> 23,425 | 6.6 / 7.6 | 0.80 | 0.80 |
| primary school teacher | education (1.00) | 2% | 5,568 -> 1,184 | 1.2 / 1.3 | 0.70 | 0.70 |
| warehouse forklift operator | logistics_transport (0.99) | 1% | 14,088 -> 486 | 2.3 / 2.2 | 0.90 | 0.90 |

**Mean over 18 queries:** 18% of the documents searched, candidates reduced by 67%, top-10 overlap with the full search 0.88, share of the full top-10 that lies inside the routed clusters 0.88. Overlap below 1 means routing LOST some results the full search returned (they live in clusters that were not searched) - this is the price of the smaller search space; whether the lost results were relevant is a Phase-4 question.
- PASS - routing returns clusters and shrinks the search space for every query
- PASS - routing never returns a document from a cluster that was not chosen (checked 18 queries x 10 results)

## 3. MapReduce (skill analysis and inverted index) equals SQL / the real index

- PASS - MapReduce skill counts are identical to SQL GROUP BY (12,414 (skill,type) counts over 60,808 jobs)
- PASS - 2 worker processes (spawn) give the same output as 1 worker (78,041 keys)
- map/shuffle/reduce stats (1 worker, all jobs): `{'records': 60808, 'chunks': 31, 'workers': 1, 'reducers': 1, 'pairs_emitted': 3362062, 'pairs_after_combine': 462583, 'keys': 146440, 'map_s': 4.519, 'shuffle_s': 0.622, 'reduce_s': 0.077, 'total_s': 5.218}`; the combiner removed 86% of the emitted pairs before the shuffle
- PASS - inverted index built by MapReduce equals the real index (first 5,000 documents) (35,323 terms)

## 4. Sequential vs parallel retrieval (first stage: candidates + scoring + merge)

Machine: 1 CPU core(s), Linux-6.18.44-fc-v64-x86_64-with-glibc2.39, Python 3.12.3, NumPy 2.4.4


**Collection x1 = 60,808 documents, model `hybrid`** (median ms per query; mean candidates scored 17,807)

| partitions P | executor | median ms | p95 ms | speedup vs sequential | efficiency | ideal speedup (estimate) | max node ms | sum node ms | merge ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sequential | 3.2 | 8.3 | 1.00x | 1.00 | 1.00x | 3.8 | 3.8 | 0.2 |
| 1 | threads | 3.3 | 8.0 | 0.99x | 0.99 | 1.00x | 3.7 | 3.7 | 0.2 |
| 1 | processes | 3.7 | 8.5 | 0.87x | 0.87 | 1.00x | 3.9 | 3.9 | 0.3 |
| 2 | sequential | 3.5 | 8.1 | 1.00x | 0.50 | 1.66x | 2.3 | 3.9 | 0.2 |
| 2 | threads | 3.3 | 8.3 | 1.06x | 0.53 | 1.66x | 3.4 | 5.3 | 0.3 |
| 2 | processes | 4.1 | 8.7 | 0.86x | 0.43 | 1.66x | 3.0 | 5.1 | 0.3 |
| 4 | sequential | 3.5 | 7.3 | 1.00x | 0.25 | 2.62x | 1.3 | 3.6 | 0.2 |
| 4 | threads | 2.9 | 6.6 | 1.20x | 0.30 | 2.62x | 2.7 | 5.3 | 0.2 |
| 4 | processes | 4.0 | 8.0 | 0.87x | 0.22 | 2.62x | 2.1 | 4.6 | 0.3 |

**Collection x1 = 60,808 documents, model `lm`** (median ms per query; mean candidates scored 17,807)

| partitions P | executor | median ms | p95 ms | speedup vs sequential | efficiency | ideal speedup (estimate) | max node ms | sum node ms | merge ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sequential | 2.3 | 6.2 | 1.00x | 1.00 | 1.00x | 2.9 | 2.9 | 0.0 |
| 1 | threads | 2.3 | 6.0 | 1.03x | 1.03 | 1.00x | 2.7 | 2.7 | 0.0 |
| 1 | processes | 2.7 | 7.0 | 0.87x | 0.87 | 1.00x | 3.2 | 3.2 | 0.0 |
| 2 | sequential | 2.6 | 6.3 | 1.00x | 0.50 | 1.69x | 1.8 | 3.0 | 0.0 |
| 2 | threads | 2.5 | 6.1 | 1.03x | 0.51 | 1.69x | 2.6 | 3.9 | 0.0 |
| 2 | processes | 3.1 | 6.8 | 0.84x | 0.42 | 1.69x | 2.6 | 4.2 | 0.0 |
| 4 | sequential | 2.4 | 6.2 | 1.00x | 0.25 | 2.83x | 1.0 | 2.9 | 0.0 |
| 4 | threads | 2.0 | 5.0 | 1.19x | 0.30 | 2.83x | 1.7 | 3.7 | 0.0 |
| 4 | processes | 3.0 | 6.9 | 0.81x | 0.20 | 2.83x | 1.9 | 4.0 | 0.0 |

*Speedup/efficiency are MEASURED wall-clock. 'Ideal speedup' is an ESTIMATE from the measured per-node times (what P free cores would give if communication were free).* This machine has 1 core(s); with fewer cores than partitions the measured speedup cannot exceed ~1 - run `python scripts/benchmark_parallel.py` on your own machine for the real figure. `docs/parallel_benchmark.md` also covers collections 4x larger.
- PASS - every executor returned the same candidate counts

**Result: ALL CHECKS PASSED**