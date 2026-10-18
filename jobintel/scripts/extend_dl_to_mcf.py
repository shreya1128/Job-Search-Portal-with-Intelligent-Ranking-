#!/usr/bin/env python3
"""Phase 5 extension: give the MyCareersFuture jobs deep-learning skill embeddings and measure how well the Phase 5 autoencoders TRANSFER to them.

Why: the Phase 5 session could not download MCF, so the autoencoders were trained / evaluated without it and `embeddings_skills.npz` has no MCF rows.  Nothing is retrained here:
  * the shipped denoising autoencoder encodes every MCF job (skills mapped to the saved vocabulary BY NAME - skill ids are not portable between databases);
  * the new rows are APPENDED to data/dl/embeddings_skills.npz (the original rows are untouched; `origin` marks them 'base' / 'mcf'; the original file is kept as embeddings_skills.base.npz);
  * zero-shot transfer test on MCF: employer-split MCF jobs, 30% of skills dropped + up to 3 random added, recall@k of the dropped skills for the shipped models vs random / popularity /
    co-occurrence baselines fitted on the MCF TRAIN jobs (the baselines therefore have an in-domain advantage), employer-clustered paired bootstrap, k-NN category purity of the latent vs raw skills.
Output: data/dl/experiments/autoencoder_mcf_transfer.json.   Run:  python scripts/extend_dl_to_mcf.py"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import keras
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sqlalchemy import text

from jobintel.db.session import get_engine
from jobintel.dl import autoencoder as A

DL = ROOT / "data" / "dl"
SQL = """SELECT j.job_id, s.name AS source, j.external_id, j.company_id, j.category, sk.name AS skill
         FROM jobs j JOIN crawler_sources s USING(source_id) JOIN job_skills js ON js.job_id = j.job_id JOIN skills sk ON sk.skill_id = js.skill_id
         WHERE s.name = 'mcf' AND j.duplicate_of IS NULL AND j.is_fraudulent = 0 AND sk.ambiguous = 0"""

if __name__ == "__main__":
    eng = get_engine()
    vocab = json.load(open(DL / "ae_vocab.json"))["vocab"]
    col = {v["name"]: i for i, v in enumerate(vocab)}
    pairs = pd.read_sql(text(SQL), eng)
    n_pairs_all = len(pairs)
    pairs = pairs[pairs.skill.isin(col)]
    print(f"MCF skill pairs: {n_pairs_all:,}; {len(pairs):,} ({100 * len(pairs) / max(n_pairs_all, 1):.1f}%) are in the saved {len(col)}-skill vocabulary (matched by name)")
    meta = pairs.drop_duplicates("job_id")[["job_id", "source", "external_id", "company_id", "category"]].reset_index(drop=True)
    row = {j: i for i, j in enumerate(meta.job_id)}
    X = sp.csr_matrix((np.ones(len(pairs), np.float32), (pairs.job_id.map(row).to_numpy(), pairs.skill.map(col).to_numpy())), shape=(len(meta), len(col)))
    X.data[:] = 1.0
    n_in = np.asarray(X.sum(axis=1)).ravel()
    models = {"denoising": keras.saving.load_model(DL / "models" / "ae_denoising.keras", compile=False), "basic": keras.saving.load_model(DL / "models" / "ae_basic.keras", compile=False)}

    # ---- 1. embeddings for every MCF job with >= 1 vocabulary skill (appended; originals untouched)
    base = DL / "embeddings_skills.npz"
    backup = DL / "embeddings_skills.base.npz"
    if not backup.exists():
        shutil.copy(base, backup)
    old = np.load(backup, allow_pickle=False)
    if "mcf" in set(old["source"].tolist()):
        sys.exit("embeddings_skills.npz already contains MCF rows; nothing to do (delete the file and restore embeddings_skills.base.npz to redo)")
    enc = A.encoder_of(models["denoising"])
    t0 = time.time()
    Z = np.vstack([enc.predict(X[i:i + 2048].toarray(), verbose=0) for i in range(0, X.shape[0], 2048)])
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    np.savez_compressed(base, source=np.concatenate([old["source"], meta.source.to_numpy(str)]), external_id=np.concatenate([old["external_id"], meta.external_id.to_numpy(str)]),
                        vectors=np.concatenate([old["vectors"], Z.astype(np.float16)]), n_skills=np.concatenate([old["n_skills"], n_in.astype(np.int16)]),
                        origin=np.concatenate([np.full(len(old["source"]), "base"), np.full(len(meta), "mcf")]))
    print(f"embeddings: kept {len(old['source']):,} original rows, appended {len(meta):,} MCF rows ({time.time() - t0:.0f}s)")

    # ---- 2. zero-shot transfer evaluation on MCF jobs with >= 3 vocabulary skills, split BY EMPLOYER
    keep = n_in >= 3
    Xk, mk = X[keep].toarray().astype(np.float32), meta[keep].reset_index(drop=True)
    grp = np.where(mk.company_id.notna(), mk.company_id.fillna(-1).astype(int), -mk.job_id.to_numpy())
    mk["split"] = A.group_split(grp, seed=42)
    tr, te = np.flatnonzero(mk.split == "train"), np.flatnonzero(mk.split == "test")
    Xtr, Xte = Xk[tr], Xk[te]
    noisy, dropped = A.corrupt(Xte, 0.3, 3, np.random.default_rng(123))
    out = {"label": "ZERO-SHOT TRANSFER: Phase 5 autoencoders (trained WITHOUT MCF) evaluated on MCF jobs; baselines are fitted on the MCF train split (in-domain advantage)",
           "n_mcf_jobs_embedded": int(len(meta)), "vocab_coverage_of_mcf_skill_pairs": float(len(pairs) / max(n_pairs_all, 1)), "n_eval_jobs": {"train": int(len(tr)), "test": int(len(te))},
           "noise": {"drop": 0.3, "add_max": 3}, "models": {}, "baselines": {}}
    scores = {}
    for name, m in models.items():
        rec = m.predict(noisy, batch_size=1024, verbose=0)
        scores[name] = rec
        out["models"][name] = {"recovery": A.recovery_at_k(rec, noisy, dropped), "spurious_skill_auc": A.spurious_detection_auc(rec, noisy, Xte), "bce_noisy_to_clean": A.bce(rec, Xte)}
    for b in ("random", "popularity", "cooc"):
        bs = A.baseline_scores(Xtr, noisy, b, np.random.default_rng(123))
        scores[b] = bs
        out["baselines"][b] = {"recovery": A.recovery_at_k(bs, noisy, dropped)}
    g_te = grp[te]
    r10 = {k: A.recall_per_job(v, noisy, dropped, 10) for k, v in scores.items()}
    for name in models:
        out["models"][name]["paired_recall@10_vs_cooc"] = A.paired_cluster_bootstrap(r10[name], r10["cooc"], g_te)
        out["models"][name]["paired_recall@10_vs_popularity"] = A.paired_cluster_bootstrap(r10[name], r10["popularity"], g_te)
    # category purity on MCF (41 categories; vague ones removed)
    lab = mk[(mk.category.notna()) & (~mk.category.isin(["Others", "General Work"]))]
    cnt = lab.category.value_counts()
    lab = lab[lab.category.isin(cnt[cnt >= 100].index)]
    ltr, lte = lab[lab.split == "train"].index.to_numpy(), lab[lab.split == "test"].index.to_numpy()
    if len(ltr) > 200 and len(lte) > 50:
        ytr, yte = lab.loc[ltr, "category"].to_numpy(), lab.loc[lte, "category"].to_numpy()
        pur = {"labelled_source": "mcf", "n_categories": int(lab.category.nunique()), "k": 10, "raw_skill_space": A.knn_purity(Xk[lte], yte, Xk[ltr], ytr, 10)}
        for name, m in models.items():
            e = A.encoder_of(m)
            pur[name] = A.knn_purity(e.predict(Xk[lte], verbose=0), yte, e.predict(Xk[ltr], verbose=0), ytr, 10)
        out["knn_category_purity"] = pur
    (DL / "experiments").mkdir(exist_ok=True)
    json.dump(out, open(DL / "experiments" / "autoencoder_mcf_transfer.json", "w"), indent=1)
    f = lambda d: round(d["recovery"]["recall@10"], 4)
    print("recall@10 of dropped skills on MCF test jobs:", {**{k: f(v) for k, v in out["models"].items()}, **{k: f(v) for k, v in out["baselines"].items()}})
    for n in models:
        p = out["models"][n]["paired_recall@10_vs_cooc"]
        print(f"  {n} - cooc: {p['mean_diff']:+.4f}  95% [{p['ci95_low']:+.4f}, {p['ci95_high']:+.4f}]  ({p['n_jobs']} jobs, {p['n_employer_groups']} employers)")
    print("purity:", json.dumps(out.get("knn_category_purity", {}))[:500])
