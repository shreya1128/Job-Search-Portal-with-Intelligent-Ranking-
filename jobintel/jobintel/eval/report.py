"""Markdown report.  Every sentence about a comparison is generated from the computed interval, so the text cannot claim more than the data show."""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Sequence

import pandas as pd

from jobintel.eval.metrics import holm
from jobintel.eval.runner import METRICS, compare, summarize
from jobintel.eval.systems import MAIN_SYSTEMS

PAIRS = [("lm_dirichlet", "keyword_baseline", "language model vs baseline"), ("dfr_pl2", "keyword_baseline", "DFR vs baseline"), ("hybrid", "keyword_baseline", "hybrid vs baseline"),
         ("hybrid", "lm_dirichlet", "hybrid vs language model"), ("hybrid", "dfr_pl2", "hybrid vs DFR"), ("lm_dirichlet", "dfr_pl2", "language model vs DFR"),
         ("lm_bigram", "lm_dirichlet", "bigram vs unigram language model"), ("hybrid", "hybrid_no_pagerank", "PageRank effect (hybrid vs hybrid without PageRank)"),
         ("hybrid", "hybrid_text_only", "skill/location/experience/PageRank signals (hybrid vs text-only hybrid)"), ("hybrid", "hybrid_equal_weights", "default weights vs equal weights"),
         ("lm_jm", "lm_dirichlet", "Jelinek-Mercer vs Dirichlet smoothing"), ("dfr_inl2", "dfr_pl2", "InL2 vs PL2"), ("lm_dirichlet_mu2000", "lm_dirichlet", "Dirichlet mu=2000 vs mu=200")]


def verdict(r: Dict, p_adj: float) -> str:
    if r is None:
        return "not computable"
    d, lo, hi, p = r["mean_diff"], r["ci_low"], r["ci_high"], r["p_value"]
    base = f"{d:+.3f} (95% CI [{lo:+.3f}, {hi:+.3f}]; p = {p:.3f}, Holm-adjusted {p_adj:.3f})"
    if p_adj < 0.05:
        return ("higher: " if d > 0 else "lower: ") + base + " - supported after correcting for the number of comparisons"
    if p < 0.05:
        return base + " - nominally p < 0.05 but NOT supported after correcting for the number of comparisons"
    return base + " - no reliable difference with these queries"


JUDGED_MIN = 0.95      # below this share of judged top-10 results a system's metrics are biased by the pooling method


def render(df: pd.DataFrame, queries: Sequence[Dict], label_stats: Dict, relevance_source: str, thr: int, depth: int, run_label: str, n_docs: int) -> str:
    names = list(dict.fromkeys(df.system))
    main = [n for n in MAIN_SYSTEMS if n in names]
    S = summarize(df, names)
    cov = df.groupby("system")["judged@10"].mean().to_dict()
    weak = {n for n, v in cov.items() if v < JUDGED_MIN}
    fmt = lambda row, m: f"{row[m]:.3f} [{row[m + '_lo']:.3f}, {row[m + '_hi']:.3f}]"
    nq = int(df[df.system == names[0]]["P@10"].notna().sum())
    out = [f"# Phase 4 - IR evaluation report ({run_label})\n",
           f"*Generated {dt.datetime.now():%Y-%m-%d %H:%M}.  Relevance source: **{relevance_source}**.  Collection: {n_docs:,} jobs.  Queries evaluated: {nq}.  Binary relevance = grade >= {thr}.  Pool depth: {depth}.*\n",
           "## How to read this report\n",
           f"* Every number is computed from the systems' ACTUAL ranked results and the relevance labels in the judgment sheet - nothing is typed in.\n"
           f"* Labels: {label_stats['judged']:,} judged jobs (grade 2: {label_stats['grade_counts'][2]:,}, grade 1: {label_stats['grade_counts'][1]:,}, grade 0: {label_stats['grade_counts'][0]:,}); "
           f"{label_stats['unlabeled']:,} pooled jobs unlabeled.\n"
           "* Brackets in sections 1-2 are 95% bootstrap intervals over the queries; with about 20 queries they are WIDE, and bootstrap intervals are themselves a little optimistic at this size. Whether two systems differ is decided only in section 3.\n"
           "* Recall is measured against the relevant jobs found in the judged pool, not against all relevant jobs in the collection (unknown).\n"
           "* Jobs no pooled system returned were never judged and count as not relevant. `judged@10` shows how much of each system's top 10 carries a judgment; a system with a low value is penalised by the method, not necessarily by its quality.\n",
           "## 1. The comparison asked for: baseline vs language model vs DFR vs hybrid IR\n",
           "| system | P@5 | P@10 | R@5 | R@10 | MRR | nDCG@10 |\n|---|---|---|---|---|---|---|"]
    for n in main:
        r = S[S.system == n].iloc[0]
        out.append(f"| **{n}** | " + " | ".join(fmt(r, m) for m in METRICS) + " |")
    if weak:
        out.append("\n> **(!) Unreliable systems:** " + ", ".join(f"{n} (only {100 * cov[n]:.0f}% of its top-10 judged)" for n in sorted(weak)) +
                   ". Their unjudged results count as NOT relevant, which lowers their scores for a reason that has nothing to do with quality (see the null self-test in docs/EVALUATION.md). "
                   "No comparison involving them is reported as supported. Label the missing jobs with `scripts/make_judgment_sheet.py --extend <labeled sheet> --systems <name>` and evaluate again.\n")
    out += ["\n## 2. All systems and variants\n", "| system | P@10 | MRR | nDCG@10 | judged@10 | median ms | queries |\n|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        out.append(f"| {r['system']}{' (!)' if r['system'] in weak else ''} | {fmt(r, 'P@10')} | {fmt(r, 'MRR')} | {fmt(r, 'nDCG@10')} | {r['judged@10']:.2f} | {r['median_ms']:.1f} | {r['queries']} |")
    results = []                                                  # (title, metric, result) for every comparison shown
    for a, b, title in PAIRS:
        if a in names and b in names:
            for m, r in compare(df, a, b).items():
                results.append((title, m, r, a in weak or b in weak))
    adj = {}
    for m in ("P@10", "MRR", "nDCG@10"):                          # Holm correction within each metric over all VALID comparisons shown
        idx = [i for i, (_, mm, r, bad) in enumerate(results) if mm == m and r and not bad]
        for i, a_ in zip(idx, holm([results[i][2]["p_value"] for i in idx])):
            adj[i] = a_
    out += ["\n## 3. Paired comparisons (difference = first minus second, same queries)\n",
            f"A difference is called supported only if the sign-flip randomization test stays below 0.05 AFTER Holm correction over the {len({i for i, x in enumerate(results) if not x[3]}) // 3} valid comparisons per metric; "
            "the bootstrap interval only describes how large the difference may be. With this many comparisons some nominal p < 0.05 are expected by chance alone.\n",
            "| comparison | metric | result | wins / ties / losses |\n|---|---|---|---|"]
    for i, (title, m, r, bad) in enumerate(results):
        out.append(f"| {title} | {m} | {'UNRELIABLE - a compared system has unjudged results; no conclusion drawn' if bad else verdict(r, adj.get(i, 1.0))} | " + (f"{r['wins']} / {r['ties']} / {r['losses']}" if r else "-") + " |")
    out += ["\n## 4. Per query (nDCG@10 / P@10)\n", "| query | relevant judged | " + " | ".join(main) + " |\n|---|---|" + "---|" * len(main)]
    for q in queries:
        sub = df[df.query_id == q["id"]]
        if sub.empty:
            continue
        cells = []
        for n in main:
            r = sub[sub.system == n]
            cells.append("-" if r.empty else ("n/a" if pd.isna(r.iloc[0]["nDCG@10"]) else f"{r.iloc[0]['nDCG@10']:.2f} / {r.iloc[0]['P@10']:.1f}"))
        out.append(f"| {q['text'][:48]} | {int(sub.iloc[0]['n_relevant_judged'])} | " + " | ".join(cells) + " |")
    out += ["\n## 5. Limitations\n",
            "* Small query set and ONE labeller (no inter-annotator agreement was measured).",
            "* Pooling bias: only the top results of the pooled systems were judged; recall and nDCG are relative to that pool.",
            "* The systems were not tuned on these queries (default parameters), which is what lets the comparison be read as a comparison; any later tuning must use different queries.",
            "* The hybrid's skill / location / experience signals use information that the pure text models ignore - the ablations in section 3 separate those effects from the text models themselves.",
            "* Queries without any relevant judged job are excluded from recall, MRR and nDCG."]
    return "\n".join(out)
