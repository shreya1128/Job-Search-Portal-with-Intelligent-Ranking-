#!/usr/bin/env python3
"""Skill-space DL embeddings for EVERY indexed dataset job (MyCareersFuture included) from the v2 denoising autoencoder (trained on a database that includes MCF).

    python scripts/build_dl_embeddings_v2.py      -> data/dl/embeddings_skills_v2.npz  (source, external_id, vectors float16 L2-normalised, n_skills)
Same format and key convention as embeddings_skills.npz (keyed by (source, external_id)); the Phase 5 file is not touched."""
import json
import os
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

if __name__ == "__main__":
    d = ROOT / "data" / "dl"
    vocab = json.load(open(d / "ae_vocab_v2.json"))["vocab"]
    col = {v["skill_id"]: i for i, v in enumerate(vocab)}
    q = ("SELECT j.job_id, s.name AS source, j.external_id, js.skill_id FROM jobs j JOIN crawler_sources s USING(source_id) JOIN job_skills js ON js.job_id = j.job_id "
         "WHERE s.source_type = 'dataset' AND j.duplicate_of IS NULL AND j.is_fraudulent = 0")
    pairs = pd.read_sql(text(q), get_engine())
    pairs = pairs[pairs.skill_id.isin(col)]
    jobs = pairs.drop_duplicates("job_id")[["job_id", "source", "external_id"]].reset_index(drop=True)
    row = {j: i for i, j in enumerate(jobs.job_id)}
    X = sp.csr_matrix((np.ones(len(pairs), np.float32), (pairs.job_id.map(row), pairs.skill_id.map(col))), shape=(len(jobs), len(col)))
    X.data[:] = 1.0
    enc = A.encoder_of(keras.saving.load_model(d / "models" / "ae_denoising_v2.keras", compile=False))
    t0 = time.time()
    Z = np.vstack([enc.predict(X[i:i + 2048].toarray(), verbose=0) for i in range(0, X.shape[0], 2048)])
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    np.savez_compressed(d / "embeddings_skills_v2.npz", source=jobs.source.to_numpy(str), external_id=jobs.external_id.to_numpy(str), vectors=Z.astype(np.float16),
                        n_skills=np.asarray(X.sum(axis=1)).ravel().astype(np.int16))
    print(f"v2 skill embeddings: {len(jobs):,} jobs x {Z.shape[1]} ({time.time() - t0:.0f}s); by source: {jobs.source.value_counts().to_dict()}")
