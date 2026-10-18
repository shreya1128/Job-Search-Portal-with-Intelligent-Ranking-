"""DL similarity signal: how close a user is to a job in the representation spaces learned in Phase 5.

Nothing runs per search except ONE forward pass of a small network for the USER (jobs were embedded once by ``scripts/build_dl_embeddings.py``).  No model is trained here.

Two spaces exist and they are NEVER mixed (a vector from one space is never compared with a vector of the other):

  text space   the recurrent representation (64-d, best sequence model by validation loss) of a job DESCRIPTION; only jobs with full text have one
               (in the offline dataset: the EMSCAD postings).  The query is the user's free text (a resume); it is used only when the profile carries a document of at
               least ``min_text_words`` words - a list of skill names is not the kind of text the network was trained on.
  skill space  the latent code (64-d) of the denoising autoencoder over the 1,500-skill multi-hot vector; every job with at least one vocabulary skill has one.
               The query is the user's skill set pushed through the same encoder.

For one job the space is chosen by the rule  text space if the query has text AND the job has a text vector, otherwise skill space.  Cosines of different spaces have different
scales, so each cosine is converted to a PERCENTILE RANK among all vectors of ITS OWN space before it is used as the signal ("this job is more similar to the user than 93% of
the jobs in its space").  The hybrid then min-max normalises over its candidates as for every other signal.

A user / job without evidence in any space gets NaN (the hybrid treats that as 'no DL evidence'); with no space at all ``available`` is False, ``score`` returns None and
the hybrid drops the signal and reports ``reason`` - no placeholder similarity is ever produced.

The older single-space entry point ``from_arrays(encoder, job_ids, embeddings)`` is kept (raw cosine against the user's text; used by the unit tests and by anyone with their own embeddings).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

MIN_TEXT_WORDS = 30


def _unit(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float32)
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.maximum(n, 1e-12)


@dataclass
class Space:
    name: str                                   # 'text' | 'skill'
    E: np.ndarray                               # (rows, d) L2-normalised
    pos: Dict[int, int]                         # job_id -> row
    zero: np.ndarray                            # rows whose vector was all zeros (no usable evidence)
    description: str = ""

    def percentile_scores(self, q: np.ndarray, job_ids: Sequence[int]) -> np.ndarray:
        """percentile rank (0..1) of cos(q, job) among the cosines of ALL rows of this space; NaN where the job has no (usable) row."""
        sims = self.E @ q
        valid = ~self.zero
        ref = np.sort(sims[valid])
        out = np.full(len(job_ids), np.nan)
        for k, j in enumerate(job_ids):
            r = self.pos.get(int(j))
            if r is not None and valid[r]:
                out[k] = float(np.searchsorted(ref, sims[r], side="right")) / max(len(ref), 1)
        return out


class DLSimilarity:
    def __init__(self, encoder=None, job_ids: Optional[np.ndarray] = None, embeddings: Optional[np.ndarray] = None, reason: Optional[str] = None, *,
                 spaces: Optional[Dict[str, Space]] = None, text_query: Optional[Callable[[str], Optional[np.ndarray]]] = None,
                 skill_query: Optional[Callable[[Dict[str, float]], Optional[np.ndarray]]] = None, min_text_words: int = MIN_TEXT_WORDS, info: Optional[Dict] = None):
        self.encoder, self.reason = encoder, reason
        self.spaces = spaces or {}
        self._text_query, self._skill_query, self.min_text_words = text_query, skill_query, min_text_words
        self.info = info or {}
        self._pos = None
        self.E = None
        if self.spaces:
            self.reason = None
        elif encoder is not None and job_ids is not None and embeddings is not None:      # legacy single space
            self.E = _unit(embeddings)
            self._pos = {int(j): i for i, j in enumerate(job_ids)}
            self.reason = None
        elif self.reason is None:
            self.reason = "no encoder / job embeddings supplied"

    # ------------------------------------------------------------------------------------------------ construction
    @property
    def available(self) -> bool:
        return bool(self.spaces) or self.E is not None

    @classmethod
    def from_arrays(cls, encoder, job_ids, embeddings) -> "DLSimilarity":
        return cls(encoder, np.asarray(job_ids), np.asarray(embeddings))

    @classmethod
    def load_default(cls, data_dir: Path, index=None, *, text_model: Optional[str] = None, use_text: bool = True, use_skills: bool = True) -> "DLSimilarity":
        """Wire the Phase 5 artefacts: ``data/dl/embeddings_text.npz`` + the best sequence model, ``embeddings_skills.npz`` + ``ae_denoising.keras`` + ``ae_vocab.json``.

        ``index`` is the loaded InvertedIndex: the embedding files are keyed by (source, external_id) (job ids differ between databases), the hybrid works with the index's job ids,
        so the index supplies the translation.  Without an index nothing can be matched and the signal stays off (with the reason)."""
        d = Path(data_dir) / "dl"
        f_text, f_skill = d / "embeddings_text.npz", d / "embeddings_skills.npz"
        if not f_text.exists() and not f_skill.exists():
            return cls(reason=f"job embeddings not found ({f_text} / {f_skill}); build them with scripts/build_dl_embeddings.py")
        if index is None:
            return cls(reason="DLSimilarity.load_default needs the loaded job index to translate (source, external_id) to job ids")
        key_to_job = {(str(s), str(e)): int(j) for j, s, e in zip(index.job_ids, index.meta["source"], index.meta["external_id"])}
        spaces: Dict[str, Space] = {}
        info: Dict = {"index_documents": int(len(key_to_job))}
        text_query = skill_query = None
        errors: List[str] = []

        def build_space(name: str, path: Path) -> Optional[Space]:
            z = np.load(path, allow_pickle=False)
            src, ext, V = z["source"], z["external_id"], np.asarray(z["vectors"], dtype=np.float32)
            norms = np.linalg.norm(V, axis=1)
            pos = {}
            for r, (s, e) in enumerate(zip(src, ext)):
                j = key_to_job.get((str(s), str(e)))
                if j is not None:
                    pos[j] = r
            info[name] = {"rows": int(len(V)), "matched_to_index": int(len(pos)), "share_of_index_documents": round(len(pos) / max(1, len(key_to_job)), 4),
                          "model": str(z["model"]) if "model" in z.files else None}
            if not pos:
                errors.append(f"{name} embeddings: no row matches a job of the index")
                return None
            return Space(name, _unit(V), pos, norms < 1e-6)

        if use_text and f_text.exists():
            sp = build_space("text", f_text)
            if sp is not None:
                try:
                    from ..dl.encoder import DLEncoder
                    enc = DLEncoder.load(d, text_model or info["text"].get("model"))
                    spaces["text"] = sp
                    text_query = lambda t, _e=enc: (lambda v: None if not np.any(v) else v)(_e.embed([t])[0])
                    info["text"]["encoder"] = enc.name
                except Exception as e:                          # missing model file / TensorFlow problem: say so, do not guess
                    errors.append(f"text space off: encoder not loadable ({type(e).__name__}: {e})")
        if use_skills and f_skill.exists() and (d / "models" / "ae_denoising.keras").exists() and (d / "ae_vocab.json").exists():
            sp = build_space("skill", f_skill)
            if sp is not None:
                try:
                    import os
                    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
                    import keras
                    from ..dl.autoencoder import encoder_of
                    vocab = json.loads((d / "ae_vocab.json").read_text(encoding="utf-8"))["vocab"]
                    col = {v["name"].lower(): i for i, v in enumerate(vocab)}
                    enc_net = encoder_of(keras.saving.load_model(d / "models" / "ae_denoising.keras", compile=False))

                    def skill_query(skills: Dict[str, float], _col=col, _net=enc_net, _n=len(vocab)):
                        x = np.zeros((1, _n), np.float32)
                        for s_ in skills:
                            i = _col.get(str(s_).lower())
                            if i is not None:
                                x[0, i] = 1.0
                        if not x.any():
                            return None
                        v = _net.predict(x, verbose=0)[0].astype(np.float32)
                        n = float(np.linalg.norm(v))
                        return v / n if n > 1e-9 else None
                    spaces["skill"] = sp
                    info["skill"]["encoder"] = "ae_denoising"
                    info["skill"]["vocabulary"] = len(vocab)
                except Exception as e:
                    errors.append(f"skill space off: autoencoder not loadable ({type(e).__name__}: {e})")
        if not spaces:
            return cls(reason="; ".join(errors) or "no usable embedding space", info=info)
        info["notes"] = errors
        return cls(spaces=spaces, text_query=text_query, skill_query=skill_query, info=info)

    # ------------------------------------------------------------------------------------------------ scoring
    def score(self, text: str, job_ids: Sequence[int], skills: Optional[Dict[str, float]] = None, document: Optional[str] = None) -> Optional[np.ndarray]:
        """one similarity per job id (NaN = this job has no evidence in the space used for it).  Returns None when the user has nothing to embed.

        single-space (legacy):  cosine(embed(text), job vector).
        two-space:              percentile rank of the cosine in the job's own space (see the module docstring); ``document`` feeds the text space, ``skills`` the skill space."""
        if not self.available:
            return None
        if not self.spaces:
            if not (text or "").strip():
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
        out = np.full(len(job_ids), np.nan, dtype=np.float64)
        used = []
        if "text" in self.spaces and self._text_query is not None and document and len(document.split()) >= self.min_text_words:
            q = self._text_query(document)
            if q is not None:
                s = self.spaces["text"].percentile_scores(q, job_ids)
                out = np.where(np.isfinite(s), s, out)
                used.append("text")
        if "skill" in self.spaces and self._skill_query is not None and skills:
            q = self._skill_query(skills)
            if q is not None:
                s = self.spaces["skill"].percentile_scores(q, job_ids)
                out = np.where(np.isfinite(out), out, s)         # only jobs the text space did not already score
                used.append("skill")
        self.last_spaces_used = used
        return out if used and np.isfinite(out).any() else None

    def explain_unavailable(self) -> str:
        return self.reason or "DL similarity not available"
