"""Training + evaluation of one model; records exactly what the syllabus table asks for: accuracy, precision, recall, F1 (macro), validation loss,
training time (plus parameters, epochs, per-class report)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def class_weights(y: np.ndarray, n: int) -> Dict[int, float]:
    cnt = np.bincount(y, minlength=n).astype(float)
    return {i: float(len(y) / (n * c)) for i, c in enumerate(cnt)}


def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray, classes) -> Dict:
    pred = proba.argmax(axis=1)
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    return dict(accuracy=float(accuracy_score(y_true, pred)), precision_macro=float(p), recall_macro=float(r), f1_macro=float(f),
                per_class=classification_report(y_true, pred, target_names=list(classes), output_dict=True, zero_division=0),
                confusion=confusion_matrix(y_true, pred, labels=range(len(classes))).tolist())


def train_one(kind: str, Xtr, ytr, Xva, yva, Xte, yte, classes, vocab_size: int, max_len: int, out_dir: Path, *, epochs: int = 8, batch: int = 64, seed: int = 42,
              patience: int = 2, verbose: int = 0, tag: Optional[str] = None, **model_kw) -> Dict:
    import keras
    from jobintel.dl.models import build_model
    keras.utils.set_random_seed(seed)
    model = build_model(kind, vocab_size, len(classes), max_len, **model_kw)
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    t0 = time.perf_counter()
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch, class_weight=class_weights(ytr, len(classes)), callbacks=[stop], verbose=verbose)
    train_s = time.perf_counter() - t0
    val_loss, val_acc = model.evaluate(Xva, yva, verbose=0)
    t1 = time.perf_counter()
    proba = model.predict(Xte, batch_size=256, verbose=0)
    infer_ms = (time.perf_counter() - t1) / len(Xte) * 1000
    res = evaluate_predictions(yte, proba, classes)
    res.update(model=kind, val_loss=float(val_loss), val_accuracy=float(val_acc), train_seconds=float(train_s), epochs_run=len(hist.history["loss"]), n_parameters=int(model.count_params()),
               history={k: [float(x) for x in v] for k, v in hist.history.items()}, inference_ms_per_job=float(infer_ms), seed=seed,
               config=dict(epochs=epochs, batch=batch, patience=patience, vocab_size=vocab_size, max_len=max_len, **model_kw))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = tag or f"{kind}_s{seed}"
    res["artifact_path"] = str(out_dir / f"{name}.keras")
    model.save(out_dir / f"{name}.keras")
    return res
