#!/usr/bin/env python3
"""Generate SYNTHETIC user interactions from the real job catalogue (see jobintel/rec/interactions.py for the generative model and its assumptions).

    python scripts/generate_synthetic_interactions.py --users 300 --seed 42                # -> data/rec/synthetic_interactions.npz (+ .json summary), no DB writes
    python scripts/generate_synthetic_interactions.py --users 300 --write-db               # also inserts users / job_interactions flagged is_synthetic = 1

The output is NOT real behaviour and must always be reported as synthetic."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import DATA_DIR
from jobintel.ir.index import InvertedIndex
from jobintel.rec.catalog import ItemCatalog
from jobintel.rec.interactions import SyntheticConfig, generate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--index", default=str(DATA_DIR / "index" / "job_index.pkl"))
    ap.add_argument("--write-db", action="store_true")
    a = ap.parse_args()
    cat = ItemCatalog(InvertedIndex.load(Path(a.index)))
    data = generate(cat, SyntheticConfig(n_users=a.users, seed=a.seed))
    out = DATA_DIR / "rec"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "synthetic_interactions.npz", user=data.user, doc=data.doc, viewed=data.viewed, clicked=data.clicked, saved=data.saved, applied=data.applied,
                        job_id=cat.ix.job_ids[data.doc])
    (out / "synthetic_interactions.json").write_text(json.dumps({**data.summary(), "personas": [{"user": p["user"], "skills": list(p["skills"]), "city": p["city"]} for p in data.personas]}, indent=1))
    print(json.dumps(data.summary(), indent=1))
    if a.write_db:
        from sqlalchemy.orm import Session
        from jobintel.db.session import get_engine
        from jobintel.rec import store
        with Session(get_engine()) as s:
            uids = store.write_synthetic(s, data, cat)
            s.commit()
        print(f"wrote {len(uids)} SYNTHETIC users and {data.n_pairs} synthetic job_interactions (is_synthetic = 1)")
