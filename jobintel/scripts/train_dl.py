#!/usr/bin/env python3
"""Train and evaluate the sequence models on the job-function classification task.

   python scripts/train_dl.py --prepare                 # build the leakage-free dataset + tokenizer (once)
   python scripts/train_dl.py --model gru               # one model          (rnn | birnn | lstm | gru | baseline | all)
   python scripts/train_dl.py --model all --epochs 8
Results: data/dl/results/<model>.json, models in data/dl/models/, rows in model_experiments."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import DATA_DIR
from jobintel.dl import data as D
from jobintel.dl.tokenizer import DLTokenizer

DL = DATA_DIR / "dl"


def prepare(seed, max_len, vocab_size):
    from jobintel.db.session import get_engine
    df, info = D.build_classification_dataset(get_engine(), seed=seed)
    D.save_dataset(df, info, DL)
    tok = DLTokenizer.fit(df[df.split == "train"].text, vocab_size=vocab_size, max_len=max_len)          # vocabulary from TRAINING text only
    tok.save(DL / "tokenizer.json")
    print(json.dumps({k: v for k, v in info.items() if k != "split_class_share"}, indent=1))
    print(f"tokenizer: {tok.vocab_size:,} words, max_len {max_len}  -> {DL / 'tokenizer.json'}")


def arrays(df, tok, classes):
    cls = {c: i for i, c in enumerate(classes)}
    out = {}
    cache = DL / "encoded.npz"
    key = f"{len(df)}-{tok.vocab_size}-{tok.max_len}"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            return {s: (z[f"X_{s}"], z[f"y_{s}"], None) for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        d = df[df.split == s]
        out[s] = (tok.encode(d.text.tolist()), d.label.map(cls).to_numpy(np.int64), d.text.tolist())
    np.savez(cache, key=key, **{f"{k}_{s}": out[s][i] for s in out for k, i in (("X", 0), ("y", 1))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--model", nargs="+", default=["all"], help="rnn birnn lstm gru baseline | all")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-existing", action="store_true", help="do not retrain a (model, seed) whose result file AND saved model already exist")
    ap.add_argument("--budget", type=float, default=None, help="seconds: stop before starting a run that would not finish in time (re-run to continue)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="train every model once per seed (random initialisation is a source of variance)")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--vocab-size", type=int, default=20000)
    
    a = ap.parse_args()
    if a.prepare:
        prepare(a.seed, a.max_len, a.vocab_size)
        return
    df, info = D.load_dataset(DL)
    tok = DLTokenizer.load(DL / "tokenizer.json")
    classes = info["classes"]
    A = arrays(df, tok, classes)
    kinds = ["baseline", "rnn", "birnn", "lstm", "gru"] if a.model == ["all"] else a.model
    EST = {"rnn": 75, "birnn": 140, "lstm": 110, "gru": 110, "baseline": 50}          # rough seconds per run on one CPU core, used only for the budget
    started = time.time()
    for kind in kinds:
        for seed in ([a.seed] if kind == "baseline" else (a.seeds or [a.seed])):
            name = "baseline" if kind == "baseline" else f"{kind}_s{seed}"
            have_result = (DL / "results" / f"{name}.json").exists()
            have_model = kind == "baseline" or (DL / "models" / f"{name}.keras").exists()
            if a.skip_existing and have_result and have_model:
                print(f"{name}: already done, skipped", flush=True)
                continue
            if a.skip_existing and have_result and not have_model:
                print(f"{name}: result file exists but the model file is missing -> retraining", flush=True)
            if a.budget and time.time() - started + EST[kind] > a.budget:
                print(f"{name}: not started (time budget) - run the same command again to continue", flush=True)
                continue
            t0 = time.time()
            if kind == "baseline":
                from jobintel.dl.baseline import run_baseline
                res = run_baseline(df, classes, DL / "results")
                name = "baseline"
            else:
                from jobintel.dl.train import train_one
                res = train_one(kind, A["train"][0], A["train"][1], A["val"][0], A["val"][1], A["test"][0], A["test"][1], classes, tok.vocab_size, tok.max_len, DL / "models",
                                epochs=a.epochs, batch=a.batch, seed=seed)
                name = f"{kind}_s{seed}"
            (DL / "results").mkdir(parents=True, exist_ok=True)
            (DL / "results" / f"{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
            print(f"{name:11s} acc {res['accuracy']:.3f}  macro-P {res['precision_macro']:.3f}  macro-R {res['recall_macro']:.3f}  macro-F1 {res['f1_macro']:.3f}  "
                  f"val_loss {res.get('val_loss', float('nan')):.3f}  train {res['train_seconds']:.0f}s  epochs {res.get('epochs_run', '-')}  params {res.get('n_parameters', '-')}   [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
