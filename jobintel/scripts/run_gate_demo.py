#!/usr/bin/env python3
"""Controlled memory task: RNN vs LSTM vs GRU memory span, and LSTM gate traces (synthetic, labelled as such).
  python scripts/run_gate_demo.py          -> data/dl/experiments/gate_demo.json"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from jobintel.dl import gate_demo as G

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None, help="substrings of setting names to (re)run, e.g. medium")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="independent training seeds (reported individually)")
    a = ap.parse_args()
    settings = {"short_gap (T=30, marker anywhere)": dict(T=30, max_start=None),
                "medium_gap (T=50, marker in the first 10 steps)": dict(T=50, max_start=10),
                "long_gap (T=80, marker in the first 10 steps)": dict(T=80, max_start=10)}
    p = ROOT / "data" / "dl" / "experiments" / "gate_demo.json"
    out = json.load(open(p)) if p.exists() else {"note": "SYNTHETIC controlled memory task for explaining gates; not part of the job workflow", "settings": {}}
    for label, kw in settings.items():
        if a.only and not any(o in label for o in a.only):
            continue
        runs = [G.train_and_eval(epochs=a.epochs, seed=sd, **kw) for sd in a.seeds]
        res, models = runs[0]["result"], runs[0]["models"]
        res["seeds"] = list(a.seeds)
        res["epochs"] = a.epochs
        for kind in res["models"]:
            accs = [r["result"]["models"][kind]["test_accuracy"] for r in runs]
            res["models"][kind]["test_accuracy_by_seed"] = accs
            res["models"][kind]["test_accuracy_mean"] = float(np.mean(accs))
        X, y, gap = G.make_memory_task(200, res["T"], seed=99, max_start=kw["max_start"])
        sel_all = []                                      # gate analysis for EVERY seed's LSTM (only meaningful where it learned the task)
        for r in runs:
            sel = [G.gate_selectivity(G.trace_example(r["models"]["lstm"], X[i]), X[i]) for i in range(50)]
            sel_all.append({k: float(np.nanmean([s[k] for s in sel])) for k in sel[0] if k not in ("marker_step", "value_step")})
        res["lstm_gate_selectivity_by_seed"] = sel_all
        res["lstm_gate_selectivity_mean_over_50_sequences"] = sel_all[0]
        ex = G.trace_example(models["lstm"], X[0])
        res["lstm_example"] = {"seed": a.seeds[0], "tokens": X[0].tolist(), "marker_position": int(np.flatnonzero(X[0] == G.MARKER)[0]), "label": int(y[0]),
                               "write_i": ex["i"].mean(axis=1).round(3).tolist(), "keep_f": ex["f"].mean(axis=1).round(3).tolist(),
                               "read_o": ex["o"].mean(axis=1).round(3).tolist()}
        out["settings"][label] = res
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(p, "w"), indent=1)                 # saved after every setting
        print(f"\n{label}: gap {res['gap_range']}, chance {res['chance']:.3f}")
        for k, v in res["models"].items():
            print(f"  {k:5s} test accuracy by seed {[round(x, 3) for x in v['test_accuracy_by_seed']]}   (seed {a.seeds[0]}: by gap {v['by_gap']})")
        print("  LSTM gates, mean over units and 50 sequences: " + ", ".join(f"{k} {v:.2f}" for k, v in res["lstm_gate_selectivity_mean_over_50_sequences"].items()))
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
