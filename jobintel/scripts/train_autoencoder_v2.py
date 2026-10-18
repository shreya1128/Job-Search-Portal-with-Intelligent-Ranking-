#!/usr/bin/env python3
"""Phase 5b: the SAME training / evaluation as train_autoencoder.py, but on a database that INCLUDES the MyCareersFuture jobs (the Phase 5 session could not download MCF).

Every output carries a _v2 suffix - no Phase 5 artefact is overwritten.  Resumable: a model that already exists is not retrained; --budget stops cleanly between models (re-run to continue).

  python scripts/train_autoencoder.py                    # everything (about 10-20 minutes on a laptop CPU)
  python scripts/train_autoencoder.py --models denoising --epochs 5       # quick check

Writes data/dl/models/ae_<name>.keras, data/dl/ae_vocab.json and data/dl/experiments/autoencoder.json.
Needs the database (jobs + job_skills).  All numbers come from the held-out test split (employers never seen in training)."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPECS = {                      # name -> (architecture, regularised, corrupted training input)
    "basic": ("basic", True, False),
    "deep": ("deep", True, False),
    "denoising": ("denoising", True, True),
    "deep_noreg": ("deep", False, False),
    "denoising_noreg": ("denoising", False, True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(SPECS), choices=list(SPECS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--top-n", type=int, default=1500)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=float, default=1e9, help="seconds; do not START another model after this long (re-run to continue)")
    ap.add_argument("--eval-only", action="store_true", help="skip training: evaluate the models already saved in data/dl/models")
    a = ap.parse_args()

    import numpy as np
    from jobintel.db.session import get_engine
    from jobintel.dl import autoencoder as A

    T0 = time.time()
    data = A.build_skill_data(get_engine(), top_n=a.top_n, seed=a.seed)
    models_dir = ROOT / "data" / "dl" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    exp = ROOT / "data" / "dl" / "experiments"
    exp.mkdir(parents=True, exist_ok=True)
    sizes = data.meta.split.value_counts().to_dict()
    print(f"skill matrix {data.X.shape}, splits {sizes}, mean skills/job {data.X.nnz / data.X.shape[0]:.1f}", flush=True)
    if not a.eval_only and not (ROOT / "data" / "dl" / "ae_vocab_v2.json").exists():
        json.dump({"vocab": data.vocab, "splits": sizes, "sources": data.meta.source.value_counts().to_dict()},
                  open(ROOT / "data" / "dl" / "ae_vocab_v2.json", "w"))

    train_log = exp / "ae_training_v2.json"                       # written after EVERY model so a crash never loses finished work
    info = json.load(open(train_log)) if train_log.exists() else {}
    import keras
    trained, histories, seconds = {}, {}, {}
    for name in a.models:
        arch, reg, noisy = SPECS[name]
        path = models_dir / f"ae_{name}_v2.keras"
        if a.eval_only:
            trained[name] = keras.saving.load_model(path, compile=False)
            histories[name] = info.get(name, {}).get("history", {})
            seconds[name] = info.get(name, {}).get("seconds")
            continue
        if path.exists() and name in info and not a.eval_only:
            trained[name] = keras.saving.load_model(path, compile=False)
            histories[name], seconds[name] = info[name]["history"], info[name]["seconds"]
            print(f"{name}: already trained, skipped", flush=True)
            continue
        if time.time() - T0 > a.budget:
            print(f"{name}: not started (time budget) - run the same command again to continue", flush=True)
            sys.exit(0)
        t0 = time.time()
        m, h = A.train_autoencoder(arch, data, epochs=a.epochs, latent=a.latent, regularise=reg, noisy_training=noisy, seed=a.seed)
        seconds[name] = round(time.time() - t0, 1)
        histories[name] = h
        trained[name] = m
        m.save(path)
        info[name] = {"history": h, "seconds": seconds[name]}
        json.dump(info, open(train_log, "w"))
        print(f"{name:16s} {seconds[name]:6.0f}s  epochs {h['epochs_run']:2d}  best val loss {h['best_val_loss']:.5f}  params {h['params']:,}", flush=True)
    print("models ready; evaluating ...", flush=True)

    res = A.evaluate_models(trained, data)
    # generalisation gap: same noise protocol on a fixed train subset vs the test split (regularisation ablation)
    rng = np.random.default_rng(5)
    tr = data.idx("train")
    sub = rng.choice(tr, size=min(5000, len(tr)), replace=False)
    Xs = data.X[sub].toarray().astype(np.float32)
    ns = A.corrupt(Xs, 0.3, 3, np.random.default_rng(6))[0]
    for name, m in trained.items():
        res["models"][name]["bce_noisy_to_clean_train_subset"] = A.bce(m.predict(ns, batch_size=1024, verbose=0), Xs)
        res["models"][name]["train_seconds"] = seconds[name]
        res["models"][name]["history"] = histories[name]
    # recovery by source for the main model vs the two non-neural rankers
    if "denoising" in trained:
        Xte = data.dense("test")
        noisy, dropped = A.corrupt(Xte, 0.3, 3, np.random.default_rng(123))
        Xtr = data.dense("train")
        scores = {"denoising": trained["denoising"].predict(noisy, batch_size=1024, verbose=0),
                  "popularity": A.baseline_scores(Xtr, noisy, "popularity"), "cooc": A.baseline_scores(Xtr, noisy, "cooc")}
        src = data.meta.source.to_numpy()[data.idx("test")]
        res["recovery_by_source"] = {s: {k: A.recovery_at_k(v[src == s], noisy[src == s], dropped[src == s], (10,))["recall@10"]
                                         for k, v in scores.items()} for s in sorted(set(src))}
    res["setting"] = {"top_n_skills": a.top_n, "latent": a.latent, "seed": a.seed, "splits": sizes,
                      "training_noise": {"drop": 0.3, "add_max": 3}, "specs": {k: SPECS[k] for k in trained}}
    json.dump(res, open(exp / "autoencoder_v2.json", "w"), indent=1)

    print("\n=== recovery of DROPPED skills (test, 30% dropped + up to 3 random added) ===")
    print(f"{'method':18s} recall@5  recall@10 recall@20  hit@10")
    rows = [(k, v["recovery"]) for k, v in res["baselines"].items()] + [(k, v["recovery"]) for k, v in res["models"].items()]
    for k, r in rows:
        print(f"{k:18s} {r['recall@5']:.3f}    {r['recall@10']:.3f}    {r['recall@20']:.3f}     {r['hit@10']:.3f}")
    print(f"\ncosine(noisy, clean) = {res['cosine_noisy_vs_clean']:.3f}")
    for k, v in res["models"].items():
        print(f"  {k:16s} cosine(top-m recon, clean) {v['cosine_recon_vs_clean']:.3f} (raw probabilities {v['cosine_probabilities_vs_clean']:.3f})   BCE noisy->clean test {v['bce_noisy_to_clean']:.5f} (train subset {v['bce_noisy_to_clean_train_subset']:.5f})")
    print("\npaired recall@10 difference vs co-occurrence baseline (employer-clustered bootstrap 95% interval):")
    for k, v in res["models"].items():
        d = v["paired_recall@10_vs_cooc"]
        print(f"  {k:16s} {d['mean_diff']:+.4f}  [{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}]")
    if "knn_category_purity" in res:
        p = res["knn_category_purity"]
        print(f"\nk-NN purity@{p['k']} on {p['labelled_source']} labels, chance {p['raw']['clean_query']['chance_purity']:.3f}")
        for k in ["raw"] + list(trained):
            print(f"  {k:16s} clean {p[k]['clean_query']['purity@%d' % p['k']]:.3f}   noisy {p[k]['noisy_query']['purity@%d' % p['k']]:.3f}")
    print("wrote", exp / "autoencoder_v2.json")


if __name__ == "__main__":
    main()
