# Design notes and honest limitations

## Decisions that came from testing on the real data (each fixed a real bug)
| Finding | Decision |
|---|---|
| Same company boilerplate reused for different cities/titles was flagged as near-duplicate (103 of 600 EMSCAD rows) | A duplicate must match **title + employer + place** *and* have identical/near-identical text. Result: 11 of 600, all genuine. Employer was added after the MCF run showed 393 cross-employer merges. |
| `"R"`, `"C"`, `"Go"`, `"excel"` (verb) matched as skills in running text | single letters never matched in free text; ~30 words that are also skills (Spark, Excel, Go, Unity ...) need a capital letter and must not start a sentence; ~30 generic words that ESCO/O*NET list (`call`, `reporting`, `source` ...) are blocked |
| Frequent generic LinkedIn tags (`retail`, `strategy`, `food`) were admitted as skills | single-word dataset tags become free-text skills **only if corroborated** by O*NET/ESCO/curated list; `technical`/`tool` types are vetted, never assumed |
| `20121-3 years` (HTML stripping glued a year to a range) produced min = 20,121 and tripped the CHECK constraint | extracted experience is validated (<= 50 years, ordered range); the constraint stays as a safety net |
| MyCareersFuture `job_id` repeats ("Recruiter" x780) | external id is made unique per source; ingestion guarantees `UNIQUE(source_id, external_id)` with a bounded suffix |
| LinkedIn's 80k distinct skill tags would bloat `skills` | only tags the lexicon vouches for are stored (skills table: 101k -> 27k) |
| `job_links` rows have different endpoint columns per link type | rows are padded with NULLs; a CHECK constraint guarantees the endpoints match the link type |

## Schema choices
* `jobs` keeps `description_original`, `description_clean` and `processed_text`; duplicates are *flagged* (`duplicate_of`), never deleted.
* `knowledge_relationships` is polymorphic (skill|occupation); MySQL cannot enforce that FK, so `scripts/verify_phase1.py` checks it.
* `job_links` stores only observed or constructed edges between real rows; the "career page" node of the PageRank graph is virtual, one per company.
* Enumerations (`text_quality`, `experience_source`, `source_type`) make data provenance explicit so later phases can filter on it.

## Testing coverage and limits
* 50 tests; the database tests run on SQLite so they need no server. The real-data runs and the schema were validated on **MariaDB 10.11**, which speaks MySQL's dialect, **not on Oracle MySQL 8**. Expected to work (standard DDL, `INSERT IGNORE`, JSON/LONGTEXT/ENUM/CHECK); please run `python scripts/init_db.py` and `python scripts/verify_phase1.py` on your MySQL and tell me if anything differs.
* Extraction has **no ground truth** on the real datasets. It is checked against hand-written cases and by inspection; its precision/recall has not been measured. If you want that in the report, label ~100 sentences and I will add the evaluation.
* Finite automata recognise patterns, not meaning (see `FINITE_AUTOMATA.md`).
* The crawler was tested only against the local fixture site. The fetcher obeys robots.txt and throttles, but crawling real sites also needs their terms to allow it.


## Phase 2 findings (each came from running the engine on the real index)
| Finding | Decision |
|---|---|
| `Company Undisclosed` (1,943 MyCareersFuture postings) became one giant employer and the top PageRank company | placeholder employer names (undisclosed, confidential, unknown ...) are treated as "no company" (`ingest/common.py`); affected sources re-ingested |
| A city typed in a natural query ranked below Singapore/Seattle jobs (location weight 0.08) | default location weight 0.20; still configurable; all defaults are provisional until Phase 4 |
| "jobs in Pune" (no text terms) returned jobs with no city | with nothing to match as text, the typed city restricts the candidate set; the shortlist is ordered by PageRank |
| A query made only of unknown words returned arbitrary documents | returns no results with a warning |
| Identical postings (same title, employer, cities) filled several top slots | opt-in `collapse_duplicates`; evaluation uses raw ranks |
| 1-word passages | fragments under 8 words are folded into the neighbouring passage |
| Min-max normalisation makes scores relative to the candidate pool | documented; do not compare scores across queries |
| PageRank correlates 0.65 with employer posting volume | documented as a prominence signal; judged only by the Phase-4 experiment |


## Phase 3 findings (each came from running the code on the real index)
| Finding | Decision |
|---|---|
| **Phase 2 bug:** when a filter shrank the candidate set, the scorers added the contribution of filtered-out documents to the *last* candidate (score off by 7.3 log-units) | every scorer restricts postings to the candidate set; test that a document's score never depends on which other documents are candidates |
| `np.argpartition` picks an arbitrary subset of the documents tied at the top-k boundary: two correct searches (single-index vs partitioned) returned different documents | `top_positions` is fully deterministic (score desc, ties by document order); randomized test against brute force |
| Passage-model shortlists were ordered differently in the local and distributed engines (tie-breaking differed) | both use document order |
| Bigram language model needs a GLOBAL bigram collection count; a partition only sees its own share | two-round protocol: nodes return counts, the coordinator sums them, then scoring |
| `spawn` (Windows) re-imported the verification script in every worker and crashed | scripts that use processes are guarded by `if __name__ == "__main__":` |
| 53% of the index sat in one `other` cluster, too coarse to route | 12 clusters, `scripts/recompute_clusters.py` re-labels the database without re-ingesting |
| Cluster routing misroutes some queries (devops -> data_ai, overlap 0.20) | routing is opt-in and documented as experimental |
| One of my verification checks was vacuous (`all(True ...)`) | replaced by a real check; verification scripts are reviewed for checks that cannot fail |
| The sandbox has 1 CPU core, so parallel wall-clock speedup cannot be measured here | the benchmark reports measured numbers and a clearly labelled estimate; you run it on your own cores |


## Phase 4 findings (each from running the evaluation harness on the real index)
| Finding | Decision |
|---|---|
| The null self-test (random labels) showed the bootstrap-interval rule calling 7.4% of comparisons "significant" with 20 queries | decisions use a paired sign-flip randomization test with Holm correction (2.9% / 0.7% under the null); bootstrap intervals only describe effect size |
| Systems whose top-10 were not pooled looked "significantly worse" in 30.6% of comparisons under random labels (unjudged = not relevant) | report flags such systems (!) and refuses conclusions about them; `--extend` adds judgments for new systems |
| Database job ids differ between machines (ingestion order), so a sheet keyed by job_id would not match your database | sheets and the index use the stable (source, external_id); labels are resolved per index; unresolved labels are reported |
| LinkedIn "experience 3-10 yrs" is an approximation derived from seniority, but would look like data to a labeller | the sheet shows `level: Mid senior` for sources without stated experience |
| `judged@10` divided by 10 even when fewer than 10 results existed | divides by the number of results returned |
| Pooling 6 systems gave 589 jobs (about 3 hours of labelling) | default pool = 5 systems / 477 jobs; bigram and other variants are added later with `--extend` |
