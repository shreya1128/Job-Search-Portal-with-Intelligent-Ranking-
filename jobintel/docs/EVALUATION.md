# IR evaluation (Phase 4): how it works, how to run it, and what its own self-test showed

## What is evaluated
Systems (all use the same index): **keyword baseline**, **language model** (Dirichlet), **DFR** (PL2), **hybrid IR**; plus variants: bigram LM, Jelinek-Mercer, Dirichlet mu = 2000, DFR InL2, passage model, and hybrid ablations (**without PageRank**, text only, equal weights).
Metrics: **P@5, P@10, R@5, R@10, MRR, nDCG@10**, retrieval time. Definitions are in `jobintel/eval/metrics.py` and checked against hand-computed values in `tests/test_eval_metrics.py`.
Relevance scale (you assign it): **2** clearly relevant, **1** partially relevant, **0** not relevant. Binary metrics count grade >= 1 (use `--thr 2` for the strict version); nDCG uses the grades.
Queries: 20, each with an explicit information need (`data/eval/queries.json` - edit freely).

## Workflow
```powershell
# 1. label (the sheet is portable: jobs are identified by source + external_id, not by your database's internal ids)
#    open relevance_judgments_to_label.xlsx, read the Instructions tab, fill the yellow 'relevance' column with 0/1/2   (477 jobs, roughly 1.5-2.5 hours)
# 2. on YOUR machine, check that your own index returns no jobs the sheet does not cover:
python scripts/make_judgment_sheet.py --extend labeled.xlsx --out data/eval/extension.xlsx     # small or empty if your data equal mine; label what it lists
# 3. evaluate
python scripts/evaluate_ir.py --judgments labeled.xlsx [extension.xlsx] --run-label manual-1   # -> docs/phase4_report.md and the ir_experiments table
# optional: other systems (e.g. bigram) - extend the pool, label the new jobs, evaluate again
python scripts/make_judgment_sheet.py --extend labeled.xlsx --systems lm_bigram lm_jm dfr_inl2 hybrid_text_only --out data/eval/extension2.xlsx
```
Short on time? Label whole queries only (at least 10 recommended) and add `--skip-unlabeled-queries`. Unlabeled rows inside a labeled query are refused.

## Design decisions that protect against fooling ourselves
* **Pooling**: the top 10 of every pooled system are merged, **shuffled**, and shown without any information on which system returned what.
* **The report text is generated from the numbers.** A difference is called "supported" only if a **paired sign-flip randomization test** stays below 0.05 **after Holm correction** over all comparisons shown; bootstrap intervals only describe effect size. Otherwise it says "no reliable difference with these queries".
* **Recall** is measured against the relevant jobs found in the pool, not against all relevant jobs in the collection (unknown).
* **Systems with unjudged top-10 results are flagged (!)** and no comparison involving them is reported as supported (see the self-test below).
* Parameters are defaults, not tuned on these queries.

## Self-test of the harness (`scripts/selftest_evaluation.py`, real index, RANDOM labels)
With labels that carry no information about the systems, nothing should look "significant". 20 random labelings, 600 paired comparisons:

| group | bootstrap interval excludes 0 | sign-flip p < 0.05 | after Holm | labelings with any supported difference |
|---|---|---|---|---|
| systems whose top-10 are all judged (the pooled ones) | 7.4% | **2.9%** | **0.7%** | 1 of 20 (about 1 expected) |
| comparisons involving a system that was NOT pooled | 37.2% | 30.6% | 21.1% | 17 of 20 |

Two lessons, both built into the tool: (1) the percentile-bootstrap interval is too optimistic with 20 queries, so it is not used to decide; (2) a system whose results were never judged (JM smoothing: only 59% of its top 10 judged; InL2: 89%) looks "significantly worse" under random labels purely because unjudged jobs count as not relevant - the **pooling bias**. That is why additional systems must be added with `--extend` before they are compared.

## Limitations (also repeated in every report)
One labeller (no inter-annotator agreement), about 20 queries (wide intervals), pooling bias for systems that were not pooled, recall relative to the pool. In the pool, `hybrid` and `hybrid_no_pagerank` share 89% of their top 10, so the effect of PageRank can only be small and will be hard to detect with 20 queries; the report will state what the data show, whichever way it goes.
