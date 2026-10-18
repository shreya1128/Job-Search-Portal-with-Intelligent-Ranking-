#!/usr/bin/env python3
"""Offline check of the collaborative filter on SYNTHETIC interactions (hold-out hit rate vs random / popularity / persona oracle).

    python scripts/evaluate_recommenders.py --users 300 --seed 42 --k 10

Writes data/rec/collab_eval_SYNTHETIC.json.  This measures whether the machinery recovers the structure the generator put in; it says nothing about real users."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import DATA_DIR
from jobintel.ir.index import InvertedIndex
from jobintel.rec.catalog import ItemCatalog
from jobintel.rec.interactions import SyntheticConfig, evaluate_holdout, generate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--index", default=str(DATA_DIR / "index" / "job_index.pkl"))
    a = ap.parse_args()
    cat = ItemCatalog(InvertedIndex.load(Path(a.index)))
    data = generate(cat, SyntheticConfig(n_users=a.users, seed=a.seed))
    res = evaluate_holdout(data, cat, k=a.k, folds=a.folds, seed=a.seed)
    res["data"] = data.summary()
    out = DATA_DIR / "rec"
    out.mkdir(parents=True, exist_ok=True)
    (out / "collab_eval_SYNTHETIC.json").write_text(json.dumps(res, indent=1))
    print(res["label"])
    print(f"users evaluated: {res['users_evaluated']}  (k={res['k']}, {res['folds']} folds, catalogue {cat.N:,} jobs; random expectation {res['random_expected']:.5f})")
    for name in ("itemknn", "popularity", "persona_oracle"):
        r = res[name]
        print(f"  {name:15s} hit@{a.k} = recall@{a.k} = {r['hit_rate']:.3f} (95% CI {r['ci95'][0]:.3f}-{r['ci95'][1]:.3f}, {r['hits']} hits)  precision@{a.k} = {r['precision@k']:.4f}  coverage = {r['coverage']:.4f}")
