# Phase 1 verification report


### Row counts

| table | rows |
|---|---|
| companies | 16210 |
| crawler_sources | 5 |
| ir_experiments | 0 |
| knowledge_relationships | 164337 |
| model_experiments | 0 |
| occupations | 4075 |
| skills | 27004 |
| users | 0 |
| jobs | 64467 |
| searches | 0 |
| user_profiles | 0 |
| user_skills | 0 |
| job_interactions | 0 |
| job_links | 1477 |
| job_pagerank | 0 |
| job_passages | 0 |
| job_skills | 981088 |
| job_terms | 0 |
| recommendations | 0 |
| skill_gaps | 0 |
| search_results | 0 |

### Jobs per source

| source | type | jobs | unique | dup % | avg tokens | exp % | salary % | edu % | label % | first date | last date |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mcf | dataset | 20298 | 18733 | 7.7 | 156 | 91 | 86 | 44 | 100 | 2018-07-09 | 2019-06-18 |
| emscad | dataset | 17880 | 17042 | 4.7 | 190 | 68 | 32 | 57 | 64 |  |  |
| linkedin | dataset | 13989 | 13989 | 0.0 | 51 | 100 | 0 | 27 | 0 | 2024-01-12 | 2024-01-17 |
| naukri | dataset | 12000 | 11948 | 0.4 | 15 | 100 | 0 | 0 | 0 |  |  |
| fixture_crawl | crawler | 300 | 299 | 0.3 | 95 | 100 | 31 | 100 | 100 | 2025-01-02 | 2025-09-28 |

### Text quality

| source | text_quality | jobs |
|---|---|---|
| emscad | full | 17800 |
| emscad | none | 80 |
| fixture_crawl | full | 300 |
| linkedin | skills_only | 13989 |
| mcf | skills_only | 1 |
| mcf | preprocessed | 20297 |
| naukri | skills_only | 12000 |

### Title-keyword clusters (heuristic, NOT ground truth)

| cluster | unique jobs |
|---|---|
| other | 33086 |
| data_ai | 11315 |
| software_it | 9992 |
| finance | 4135 |
| healthcare | 3483 |

### Top technical skills (unique jobs; dataset tags + extracted)

| skill | jobs |
|---|---|
| microsoft office | 8913 |
| data analysis | 8817 |
| python | 8567 |
| excel | 7935 |
| sql | 7608 |
| java | 6732 |
| linux | 4757 |
| javascript | 4705 |
| software development | 4699 |
| machine learning | 4331 |
| html | 3602 |
| business intelligence | 3237 |
| c++ | 3164 |
| rest api | 3048 |
| aws | 3042 |

### Skill origin

| origin | links |
|---|---|
| dataset_tag | 658308 |
| extracted | 298809 |
| both | 23971 |

### Knowledge graph

| source | relation | edges |
|---|---|---|
| curated | broader_than | 157 |
| curated | requires_optional | 113 |
| curated | requires_essential | 73 |
| curated | related_to | 28 |
| esco | requires_essential | 66561 |
| esco | requires_optional | 60838 |
| esco | related_to | 5719 |
| onet | related_to | 18460 |
| onet | uses_technology | 12388 |

### Integrity checks

- PASS - jobs without a source (0)
- PASS - indexable jobs (non-duplicate, has text) missing processed_text (0)
- PASS - original description lost (full-text job with NULL description_original) (0)
- PASS - experience_min > experience_max (0)
- PASS - duplicate_of pointing to a missing job (0)
- PASS - job is its own duplicate (0)
- PASS - knowledge edge: subject does not exist (0)
- PASS - knowledge edge: object does not exist (0)
- PASS - skills linked to jobs but ambiguous single letters (0)
- PASS - database has job rows (64467)
- PASS - knowledge base populated (164337)

**Result: ALL CHECKS PASSED**