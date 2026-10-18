"""DL similarity signal (Phase 5 -> Phase 6 bridge).  Two adapters with the same interface (`available`, `reason`, `score_docs(profile, catalog, docs)`):

  DLSkillSimilarity  the production one: the profile's skills -> multi-hot over the autoencoder vocabulary -> latent code of the DENOISING autoencoder; cosine with the pre-computed
                     latent code of each job (embeddings keyed by (source, external_id): job ids are not portable between databases).  Jobs without an embedding get NaN, which the
                     hybrid turns into a NEUTRAL value (no penalty, no bonus) and reports as DL coverage.  The text-space (recurrent) embeddings are a different space and are NOT mixed in.
  DLSimilarity       generic text-encoder adapter (any object with embed(texts)), kept for the Phase-5 DLEncoder text space.

Both stay OFF (and say why) when their artefacts are missing - no placeholder similarity is ever produced.

(original note of the generic adapter follows)
DL similarity signal - an ADAPTER that stays OFF until real Phase 5 artefacts exist.

It needs (1) an encoder with `embed(texts) -> (n, d) array` (the Phase 5 DLEncoder) and (2) job embeddings aligned with job ids (Phase 5's job-embedding build).
Without both, `available` is False, `score()` returns None and the hybrid drops the signal and says why: no placeholder similarity is ever produced.
The exact Phase 5 file format / class path must be wired in `load_default` once the Phase 5 deliverables are final; `from_arrays` is the stable entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


class DLSimilarity:
    def __init__(self, encoder=None, job_ids: Optional[np.ndarray] = None, embeddings: Optional[np.ndarray] = None, reason: Optional[str] = None):
        self.encoder, self.reason = encoder, reason
        self._pos = None
        self.E = None
        if encoder is not None and job_ids is not None and embeddings is not None:
            E = np.asarray(embeddings, dtype=np.float32)
            n = np.linalg.norm(E, axis=1, keepdims=True)
            self.E = E / np.maximum(n, 1e-12)
            self._pos = {int(j): i for i, j in enumerate(job_ids)}
            self.reason = None
        elif self.reason is None:
            self.reason = "no encoder / job embeddings supplied"

    @property
    def available(self) -> bool:
        return self.E is not None

    @classmethod
    def from_arrays(cls, encoder, job_ids, embeddings) -> "DLSimilarity":
        return cls(encoder, np.asarray(job_ids), np.asarray(embeddings))

    @classmethod
    def load_default(cls, data_dir: Path) -> "DLSimilarity":
        f = Path(data_dir) / "dl" / "job_embeddings.npz"
        if not f.exists():
            return cls(reason=f"job embeddings not found ({f}); build them with the Phase 5 embedding script")
        try:
            from ..dl.encoder import DLEncoder                      # Phase 5 module (not part of the Phase 4 snapshot)
        except Exception as e:                                       # ImportError or a TensorFlow problem
            return cls(reason=f"Phase 5 encoder not importable ({type(e).__name__}: {e})")
        return cls(reason="Phase 5 wiring not finished: connect DLEncoder / the embedding file format here")

    def score_docs(self, profile, catalog, docs: Sequence[int]) -> Optional[np.ndarray]:
        return self.score(profile.text(), [int(catalog.ix.job_ids[d]) for d in docs])

    def score(self, text: str, job_ids: Sequence[int]) -> Optional[np.ndarray]:
        """cosine(embedding of `text`, job embedding) for each job id; jobs without an embedding get NaN (the caller treats them as 0)."""
        if not self.available or not text.strip():
            return None
        q = np.asarray(self.encoder.embed([text]), dtype=np.float32)[0]
        n = np.linalg.norm(q)
        if n == 0:
            return None
        q = q / n
        out = np.full(len(job_ids), np.nan, dtype=np.float64)
        for k, j in enumerate(job_ids):
            i = self._pos.get(int(j))
            if i is not None:
                out[k] = float(self.E[i] @ q)
        return out


class DLSkillSimilarity:
    """skill-space DL similarity: DAE latent(profile skills) . DAE latent(job skills)."""

    def __init__(self, encoder=None, vocab_names: Optional[Sequence[str]] = None, catalog=None, source=None, external_id=None, vectors=None, reason: Optional[str] = None, label: str = ""):
        self.encoder, self.reason, self.label = encoder, reason, label
        self.col = {str(n).lower(): i for i, n in enumerate(vocab_names)} if vocab_names is not None else {}
        self.E = None
        self.doc_row = None
        if encoder is not None and catalog is not None and vectors is not None:
            E = np.asarray(vectors, dtype=np.float32)
            E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
            row = {(str(s), str(e)): i for i, (s, e) in enumerate(zip(source, external_id))}
            m = catalog.ix.meta
            self.doc_row = np.array([row.get((str(m["source"][d]), str(m["external_id"][d])), -1) for d in range(catalog.N)], dtype=np.int64)
            self.E, self.reason = E, None
            self.n_docs_covered = int((self.doc_row >= 0).sum())
        elif self.reason is None:
            self.reason = "no encoder / embeddings supplied"

    @property
    def available(self) -> bool:
        return self.E is not None

    @classmethod
    def from_files(cls, data_dir: Path, catalog, version: str = "v2") -> "DLSkillSimilarity":
        d = Path(data_dir) / "dl"
        sfx = f"_{version}" if version else ""
        vocab_f, model_f, emb_f = d / f"ae_vocab{sfx}.json", d / "models" / f"ae_denoising{sfx}.keras", d / f"embeddings_skills{sfx}.npz"
        missing = [f.name for f in (vocab_f, model_f, emb_f) if not f.exists()]
        if missing:
            return cls(reason=f"Phase 5 artefact(s) not found in {d}: {', '.join(missing)}")
        try:
            import json
            import keras
            model = keras.saving.load_model(model_f, compile=False)
            enc = keras.Model(model.input, model.get_layer("latent").output)
            names = [v["name"] for v in json.load(open(vocab_f))["vocab"]]
            z = np.load(emb_f, allow_pickle=False)
            return cls(enc, names, catalog, z["source"], z["external_id"], z["vectors"], label=f"denoising autoencoder {version or 'v1'}")
        except Exception as e:                                           # TensorFlow / file problems must never break recommending
            return cls(reason=f"could not load the DL artefacts ({type(e).__name__}: {e})")

    def profile_vector(self, skills) -> Optional[np.ndarray]:
        x = np.zeros((1, len(self.col)), dtype=np.float32)
        for s in skills:
            i = self.col.get(str(s).lower())
            if i is not None:
                x[0, i] = 1.0
        if not x.any():
            return None
        z = np.asarray(self.encoder.predict(x, verbose=0), dtype=np.float32)[0]
        n = np.linalg.norm(z)
        return None if n < 1e-9 else z / n

    def score_docs(self, profile, catalog, docs: Sequence[int]) -> Optional[np.ndarray]:
        if not self.available:
            return None
        z = self.profile_vector(profile.skills)
        if z is None:
            return None
        rows = self.doc_row[np.asarray(docs, dtype=np.int64)]
        out = np.full(len(rows), np.nan)
        ok = rows >= 0
        out[ok] = self.E[rows[ok]] @ z
        return out
