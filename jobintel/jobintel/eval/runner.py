"""Runs every system on every query, computes the metrics from the ACTUAL results and the relevance labels, compares systems with paired
bootstrap intervals and stores everything (ir_experiments + a markdown report)."""
from __future__ import annotations

import datetime as dt
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from jobintel.eval import metrics as E

METRICS = ["P@5", "P@10", "R@5", "R@10", "MRR", "nDCG@10"]


def evaluate(systems: Dict[str, Callable], names: Sequence[str], queries: Sequence[Dict], labels: Dict[str, Dict[int, int]], thr: int = 1,
             progress: Optional[Callable] = None) -> pd.DataFrame:
    rows = []
    for name in names:
        for q in queries:
            lab = labels.get(q["id"])
            if not lab:
                continue
            ids, ms = systems[name](q["text"])
            rows.append({"system": name, "query_id": q["id"], "query": q["text"], "n_results": len(ids), "latency_ms": ms, "judged@10": E.judged_at_k(ids, lab, 10),
                         "P@5": E.precision_at_k(ids, lab, 5, thr), "P@10": E.precision_at_k(ids, lab, 10, thr), "R@5": E.recall_at_k(ids, lab, 5, thr),
                         "R@10": E.recall_at_k(ids, lab, 10, thr), "MRR": E.reciprocal_rank(ids, lab, thr), "nDCG@10": E.ndcg_at_k(ids, lab, 10),
                         "n_relevant_judged": E.n_relevant(lab, thr)})
        if progress:
            progress(name)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, names: Sequence[str], metrics: Sequence[str] = METRICS) -> pd.DataFrame:
    out = []
    for n in names:
        sub = df[df.system == n]
        row = {"system": n}
        for m in metrics:
            v = sub[m].dropna().to_numpy(float)
            ci = E.bootstrap_mean_ci(v) if len(v) else None
            row[m] = ci["mean"] if ci else np.nan
            row[m + "_lo"], row[m + "_hi"] = (ci["ci_low"], ci["ci_high"]) if ci else (np.nan, np.nan)
        row["judged@10"] = sub["judged@10"].mean()
        row["median_ms"] = sub["latency_ms"].median()
        row["queries"] = int(sub["P@10"].notna().sum())
        out.append(row)
    return pd.DataFrame(out)


def compare(df: pd.DataFrame, a: str, b: str, metrics: Sequence[str] = ("P@10", "MRR", "nDCG@10")) -> Dict[str, Dict]:
    """a minus b, paired over the queries on which the metric is defined."""
    res = {}
    for m in metrics:
        x = df[df.system == a].set_index("query_id")[m]
        y = df[df.system == b].set_index("query_id")[m]
        keep = x.dropna().index.intersection(y.dropna().index)
        res[m] = E.paired_bootstrap(x[keep].to_numpy(float), y[keep].to_numpy(float)) if len(keep) else None
    return res


def store_results(engine, df: pd.DataFrame, run_label: str, relevance_source: str, n_docs: int, config: Dict) -> int:
    from sqlalchemy import delete, insert
    from jobintel.db.models import IRExperiment
    rows = [dict(run_label=run_label, query_text=r["query"][:1000], ranking_model=r["system"], k=10, precision_at_k=r["P@10"], recall_at_k=None if pd.isna(r["R@10"]) else r["R@10"],
                 mrr=None if pd.isna(r["MRR"]) else r["MRR"], ndcg=None if pd.isna(r["nDCG@10"]) else r["nDCG@10"], latency_ms=r["latency_ms"], n_docs=n_docs,
                 execution_mode="sequential", relevance_source=relevance_source,
                 config={**config, "query_id": r["query_id"], "P@5": r["P@5"], "R@5": None if pd.isna(r["R@5"]) else r["R@5"]}) for r in df.to_dict("records")]
    with engine.begin() as c:
        c.execute(delete(IRExperiment).where(IRExperiment.run_label == run_label))
        for i in range(0, len(rows), 500):
            c.execute(insert(IRExperiment), rows[i:i + 500])
    return len(rows)
