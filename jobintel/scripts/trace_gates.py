#!/usr/bin/env python3
"""Show what the LSTM / GRU gates do on a REAL job description, computed with the from-scratch NumPy cells and checked
against the trained Keras model.

  python scripts/trace_gates.py                         # one real test-split job, LSTM and GRU (best seed by validation loss)
  python scripts/trace_gates.py --text "Senior Java developer with 5 years ..." --kind lstm
Writes data/dl/experiments/gates_example.json."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DL = ROOT / "data" / "dl"


def best_run(kind):
    runs = [json.load(open(f)) | {"name": f.stem} for f in (DL / "results").glob(f"{kind}_s*.json")
            if (DL / "models" / f"{f.stem}.keras").exists()]
    return min(runs, key=lambda r: r["val_loss"])["name"] if runs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None)
    ap.add_argument("--kind", nargs="+", default=["lstm", "gru"], choices=["lstm", "gru"])
    ap.add_argument("--tokens", type=int, default=40, help="how many leading tokens to trace")
    a = ap.parse_args()

    import keras
    import numpy as np
    import pandas as pd
    from jobintel.dl import numpy_rnn as nr
    from jobintel.dl.models import representation_model
    from jobintel.dl.tokenizer import DLTokenizer, words

    tok = DLTokenizer.load(DL / "tokenizer.json")
    if a.text:
        text, source = a.text, "command line"
    else:
        df = pd.read_csv(DL / "classification.csv")
        t = df[(df.split == "test") & (df.label == "Information Technology")]
        t = t[t.text.str.split().str.len().between(80, 200)]
        row = t.iloc[0]
        text, source = row.text, f"EMSCAD test split, job_id {row.job_id}, label {row.label}"
    toks = words(text)[: tok.max_len]
    ids = tok.encode_one(text)
    out = {"source": source, "n_tokens_total": len(ids), "traced_tokens": min(a.tokens, len(ids)), "models": {}}
    for kind in a.kind:
        name = best_run(kind)
        if name is None:
            print(f"no saved {kind} model - run scripts/train_dl.py --model {kind}")
            continue
        model = keras.saving.load_model(DL / "models" / f"{name}.keras", compile=False)
        tr = nr.trace_sentence(model, kind, ids)
        padded = np.zeros((1, tok.max_len), np.int32)
        padded[0, : len(ids)] = ids
        ref = representation_model(model).predict(padded, verbose=0)[0]
        diff = float(np.max(np.abs(ref - tr["h_last"])))
        rows = nr.gate_summary(tr, toks, kind)
        out["models"][kind] = {"run": name, "max_abs_diff_numpy_vs_keras": diff, "rows": rows[: a.tokens]}
        print(f"\n=== {kind.upper()} ({name}) - NumPy forward pass vs Keras: max |diff| = {diff:.2e}   ({'OK' if diff < 1e-4 else 'MISMATCH'})")
        keys = [k for k in rows[0] if k != "token"]
        print(f"{'#':>3} {'token':16s} " + " ".join(f"{k:>12s}" for k in keys))
        for i, r in enumerate(rows[: a.tokens]):
            print(f"{i:3d} {r['token'][:16]:16s} " + " ".join(f"{r[k]:12.3f}" for k in keys))
        col = keys[0] if kind == "lstm" else "state_change"
        order = sorted(range(len(rows)), key=lambda i: -rows[i][col if kind == "lstm" else "state_change"])[:5]
        print("largest", "write-gate" if kind == "lstm" else "state change", "at:", [(rows[i]["token"], round(rows[i][col], 3)) for i in order])
        if kind == "lstm":
            print("mean gates over the whole description: write_i %.3f  keep_f %.3f  read_o %.3f" % (
                np.mean([r["write_i"] for r in rows]), np.mean([r["keep_f"] for r in rows]), np.mean([r["read_o"] for r in rows])))
            out["models"][kind]["mean_over_all_tokens"] = {k: float(np.mean([r[k] for r in rows])) for k in ("write_i", "keep_f", "read_o")}
        else:
            out["models"][kind]["mean_over_all_tokens"] = {k: float(np.mean([r[k] for r in rows])) for k in ("update_z", "reset_r")}
    (DL / "experiments").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(DL / "experiments" / "gates_example.json", "w"), indent=1)
    print("\nwrote", DL / "experiments" / "gates_example.json")


if __name__ == "__main__":
    main()
