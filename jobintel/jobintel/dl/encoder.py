"""The deployed deep-learning component: load the best trained sequence model once and use it for
   (1) job-description embeddings (the learned representation, for the DL-similarity signal),
   (2) function/category prediction (for filters, cluster routing and the NER/resume pipeline).

Nothing here trains anything.  The model, tokenizer and class list are loaded from disk; the SAME tokenizer object used in
training is used here, so text is encoded identically."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from jobintel.config import DATA_DIR
from jobintel.dl.tokenizer import DLTokenizer

DL_DIR = Path(DATA_DIR) / "dl"


def available_runs(directory: Path = DL_DIR) -> List[Dict]:
    """Result files whose saved model exists, with their validation loss (lower = better)."""
    runs = []
    for f in sorted((directory / "results").glob("*.json")):
        r = json.loads(f.read_text())
        if f.stem == "baseline" or "val_loss" not in r:
            continue
        path = directory / "models" / f"{f.stem}.keras"
        if path.exists():
            runs.append({"name": f.stem, "kind": r.get("model"), "val_loss": r["val_loss"], "val_accuracy": r.get("val_accuracy"),
                         "path": path})
    return runs


class DLEncoder:
    def __init__(self, model, tokenizer: DLTokenizer, classes: Sequence[str], name: str):
        import keras
        from jobintel.dl.models import representation_model

        self.model, self.tokenizer, self.classes, self.name = model, tokenizer, list(classes), name
        self._rep = representation_model(model)
        self.dim = int(self._rep.output_shape[-1])

    @classmethod
    def load(cls, directory: Path = DL_DIR, name: Optional[str] = None) -> "DLEncoder":
        """Load run ``name`` (e.g. 'lstm_s43'), or by default the run with the lowest VALIDATION loss (never the test set)."""
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import keras

        runs = available_runs(directory)
        if not runs:
            raise FileNotFoundError(f"no trained model in {directory / 'models'} - run scripts/train_dl.py first")
        if name is None:
            run = min(runs, key=lambda r: r["val_loss"])
        else:
            hit = [r for r in runs if r["name"] == name]
            if not hit:
                raise FileNotFoundError(f"model {name!r} not found; available: {[r['name'] for r in runs]}")
            run = hit[0]
        info = json.loads((directory / "classification_info.json").read_text())
        return cls(keras.saving.load_model(run["path"], compile=False), DLTokenizer.load(directory / "tokenizer.json"), info["classes"], run["name"])

    # ----------------------------------------------------------------------------------------------- inference
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            raise TypeError("pass a list of texts, not a single string")
        return self.tokenizer.encode(["" if t is None else str(t) for t in texts])

    def embed(self, texts: Sequence[str], batch: int = 128) -> np.ndarray:
        """L2-normalised recurrent representation, shape (n, dim).  A text with no usable tokens gets the zero vector."""
        ids = self._encode(texts)
        if len(ids) == 0:
            return np.zeros((0, self.dim), np.float32)
        v = self._rep.predict(ids, batch_size=batch, verbose=0).astype(np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        out = np.divide(v, n, out=np.zeros_like(v), where=n > 1e-9)
        out[(ids > 0).sum(axis=1) == 0] = 0.0
        return out

    def predict_proba(self, texts: Sequence[str], batch: int = 128) -> np.ndarray:
        ids = self._encode(texts)
        if len(ids) == 0:
            return np.zeros((0, len(self.classes)), np.float32)
        return self.model.predict(ids, batch_size=batch, verbose=0)

    def predict_category(self, texts: Sequence[str], top: int = 3) -> List[Dict]:
        """Top categories with probabilities; texts with no usable tokens return {'category': None}."""
        ids = self._encode(texts)
        p = self.predict_proba(texts)
        out = []
        for row, n_tok in zip(p, (ids > 0).sum(axis=1)):
            if n_tok == 0:
                out.append({"category": None, "reason": "text has no usable tokens", "top": []})
                continue
            order = np.argsort(-row)[:top]
            out.append({"category": self.classes[int(order[0])], "probability": float(row[order[0]]),
                        "top": [{"category": self.classes[int(i)], "probability": float(row[i])} for i in order]})
        return out

    def similarity(self, text_a: str, text_b: str) -> float:
        a, b = self.embed([text_a, text_b])
        return float(a @ b)
