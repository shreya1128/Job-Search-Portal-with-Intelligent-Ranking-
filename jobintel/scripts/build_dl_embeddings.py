#!/usr/bin/env python3
"""Pre-compute the deep-learning vectors of every job ONCE, so no model runs at search time.

  python scripts/build_dl_embeddings.py                # text embeddings (best sequence model) + skill-latent (denoising AE)

Outputs (keyed by (source, external_id): job ids are not portable between databases)
  data/dl/embeddings_text.npz     jobs with full description text: L2-normalised recurrent representation (dim = hidden units)
  data/dl/embeddings_skills.npz   jobs with >= 1 vocabulary skill: L2-normalised latent code of the denoising autoencoder
The two spaces are different and are never mixed; the recommender uses the text space when a job has text, otherwise the skill space.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["text", "skills"], default=None)
    ap.add_argument("--model", default=None, help="sequence model name (default: lowest validation loss)")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N jobs")
    a = ap.parse_args()

    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from sqlalchemy import text
    from jobintel.db.session import get_engine

    eng = get_engine()
    out_dir = ROOT / "data" / "dl"

    if a.only in (None, "text"):
        from jobintel.dl.encoder import DLEncoder
        enc = DLEncoder.load(out_dir, a.model)
        q = ("SELECT j.job_id, s.name AS source, j.external_id, j.description_clean AS text FROM jobs j JOIN crawler_sources s USING(source_id) "
             "WHERE s.source_type = 'dataset' AND j.text_quality = 'full' AND j.description_clean IS NOT NULL")
        df = pd.read_sql(text(q), eng)
        if a.limit:
            df = df.head(a.limit)
        t0 = time.time()
        vec = np.vstack([enc.embed(df.text.iloc[i:i + 512].tolist()) for i in range(0, len(df), 512)]) if len(df) else np.zeros((0, enc.dim), np.float32)
        empty = float((np.linalg.norm(vec, axis=1) == 0).mean()) if len(df) else 0.0
        np.savez_compressed(out_dir / "embeddings_text.npz", source=df.source.to_numpy(str), external_id=df.external_id.to_numpy(str),
                            vectors=vec.astype(np.float16), model=enc.name)
        print(f"text embeddings: {len(df):,} jobs x {enc.dim} with {enc.name}  ({time.time() - t0:.0f}s, {100 * empty:.1f}% empty)", flush=True)
        # sanity check: do neighbours in the embedding space share the job function?  (test jobs as queries, TRAIN jobs as reference;
        # the model never saw the test split, and the split is by employer)
        cls_path = out_dir / "classification.csv"
        if cls_path.exists() and len(df) and not a.limit:
            from jobintel.dl import autoencoder as A
            lab = pd.read_csv(cls_path, usecols=["job_id", "label", "split"])
            pos = pd.Series(np.arange(len(df)), index=df.job_id.to_numpy())
            lab = lab[lab.job_id.isin(pos.index)]
            tr, te = lab[lab.split == "train"], lab[lab.split == "test"]
            if len(tr) and len(te):
                res = A.knn_purity(vec[pos[te.job_id].to_numpy()], te.label.to_numpy(), vec[pos[tr.job_id].to_numpy()], tr.label.to_numpy(), k=10)
                res["model"] = enc.name
                (out_dir / "experiments").mkdir(exist_ok=True)
                json.dump(res, open(out_dir / "experiments" / "embedding_check.json", "w"), indent=1)
                print(f"embedding check: purity@10 {res['purity@10']:.3f} (chance {res['chance_purity']:.3f}), 10-NN vote accuracy {res['knn_accuracy']:.3f}", flush=True)

    if a.only in (None, "skills"):
        import keras
        from jobintel.dl import autoencoder as A
        vocab = json.load(open(out_dir / "ae_vocab.json"))["vocab"]
        col = {v["skill_id"]: i for i, v in enumerate(vocab)}
        q = ("SELECT j.job_id, s.name AS source, j.external_id, js.skill_id FROM jobs j JOIN crawler_sources s USING(source_id) "
             "JOIN job_skills js ON js.job_id = j.job_id WHERE s.source_type = 'dataset'")
        pairs = pd.read_sql(text(q), eng)
        pairs = pairs[pairs.skill_id.isin(col)]
        jobs = pairs.drop_duplicates("job_id")[["job_id", "source", "external_id"]].reset_index(drop=True)
        if a.limit:
            jobs = jobs.head(a.limit)
        row = {j: i for i, j in enumerate(jobs.job_id)}
        pairs = pairs[pairs.job_id.isin(row)]
        X = sp.csr_matrix((np.ones(len(pairs), np.float32), (pairs.job_id.map(row), pairs.skill_id.map(col))), shape=(len(jobs), len(col)))
        X.data[:] = 1.0
        model = keras.saving.load_model(out_dir / "models" / "ae_denoising.keras", compile=False)
        enc = A.encoder_of(model)
        t0 = time.time()
        Z = np.vstack([enc.predict(X[i:i + 2048].toarray(), verbose=0) for i in range(0, X.shape[0], 2048)])
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
        np.savez_compressed(out_dir / "embeddings_skills.npz", source=jobs.source.to_numpy(str), external_id=jobs.external_id.to_numpy(str),
                            vectors=Z.astype(np.float16), n_skills=np.asarray(X.sum(axis=1)).ravel().astype(np.int16))
        print(f"skill embeddings: {len(jobs):,} jobs x {Z.shape[1]} with the denoising autoencoder ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
