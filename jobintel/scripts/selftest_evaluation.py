#!/usr/bin/env python3
"""NULL SELF-TEST of the evaluation harness on the real index with RANDOM labels (nothing is saved to the database or docs).
With random labels no system is genuinely better, so almost no paired comparison should come out 'significant'.  A harness that reports
large systematic differences here would be broken (or would be reading something other than the labels)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import DATA_DIR
from jobintel.db.session import get_engine
from jobintel.eval import judgments as J
from jobintel.eval import runner as R
from jobintel.eval.queries import load_queries
from jobintel.eval.metrics import holm
from jobintel.eval.report import PAIRS
from jobintel.eval.systems import MAIN_SYSTEMS, POOL_SYSTEMS, build_systems
from jobintel.ir.engine import DbDocStore
from jobintel.ir.index import InvertedIndex


def main():
    queries = load_queries(DATA_DIR / "eval" / "queries.json")
    ix = InvertedIndex.load(DATA_DIR / "index" / "job_index.pkl")
    systems = build_systems(ix, DbDocStore(get_engine()), k=10)
    pool = J.build_pool(systems, queries, POOL_SYSTEMS, 10)
    names = [n for n in systems if n in POOL_SYSTEMS or n in ("lm_jm", "dfr_inl2", "hybrid_text_only")]
    pooled = set(POOL_SYSTEMS)
    groups = {"pooled systems only (all top-10 judged)": lambda a, b: a in pooled and b in pooled,
              "comparisons involving a system that was NOT pooled": lambda a, b: not (a in pooled and b in pooled)}
    tally = {g: dict(n=0, ci=0, raw=0, holm=0, fam=0, judged=[]) for g in groups}
    for seed in range(20):                                        # twenty different random labelings (labels carry NO information about the systems)
        rng = np.random.default_rng(seed)
        labels = {q: {j: int(rng.choice([0, 1, 2], p=[0.6, 0.25, 0.15])) for j in p["job_ids"]} for q, p in pool.items()}
        df = R.evaluate(systems, names, queries, labels)
        for g, pick in groups.items():
            t, fam = tally[g], False
            for m in ("P@10", "MRR", "nDCG@10"):
                res = [R.compare(df, a, b, (m,))[m] for a, b, _ in PAIRS if a in names and b in names and pick(a, b)]
                res = [r for r in res if r]
                adj = holm([r["p_value"] for r in res])
                t["n"] += len(res); t["ci"] += sum(r["ci_low"] > 0 or r["ci_high"] < 0 for r in res); t["raw"] += sum(r["p_value"] < 0.05 for r in res)
                t["holm"] += sum(p_ < 0.05 for p_ in adj); fam |= any(p_ < 0.05 for p_ in adj)
            t["fam"] += fam
    cov = df.groupby("system")["judged@10"].mean()
    print("share of each system's top-10 that carries a judgment: " + ", ".join(f"{n} {cov[n]:.2f}" for n in names))
    for g, t in tally.items():
        n = t["n"]
        print(f"\n{g}: {n} paired comparisons over 20 random labelings")
        print(f"  'bootstrap interval excludes 0'   : {100 * t['ci'] / n:5.1f}%")
        print(f"  sign-flip randomization p < 0.05  : {100 * t['raw'] / n:5.1f}%   (about 5% expected when nothing is real)")
        print(f"  ... after Holm correction         : {100 * t['holm'] / n:5.1f}%")
        print(f"  labelings with ANY Holm-supported difference: {t['fam']} of 20   (about 1 of 20 expected)")


if __name__ == "__main__":
    main()
