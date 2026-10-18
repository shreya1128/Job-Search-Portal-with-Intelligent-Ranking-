"""Non-neural reference points (so a neural score can be read against something): majority class, and a bag-of-words logistic regression."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from jobintel.dl.train import evaluate_predictions


def run_baseline(df, classes, out_dir: Path):
    cls = {c: i for i, c in enumerate(classes)}
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    ytr, yva, yte = (d.label.map(cls).to_numpy() for d in (tr, va, te))
    t0 = time.perf_counter()
    vec = CountVectorizer(max_features=20000, min_df=3, binary=True, lowercase=True)
    Xtr = vec.fit_transform(tr.text)
    clf = LogisticRegression(max_iter=400, class_weight="balanced", C=0.5)
    clf.fit(Xtr, ytr)
    train_s = time.perf_counter() - t0
    proba = clf.predict_proba(vec.transform(te.text))
    from sklearn.metrics import log_loss
    res = evaluate_predictions(yte, proba, classes)
    res.update(model="baseline", val_loss=float(log_loss(yva, clf.predict_proba(vec.transform(va.text)), labels=range(len(classes)))), train_seconds=float(train_s),
               n_parameters=int(clf.coef_.size), epochs_run=None, note="binary bag-of-words + multinomial logistic regression (class-balanced); NOT a sequence model")
    maj = np.bincount(ytr).argmax()
    res["majority_class_accuracy"] = float((yte == maj).mean())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res
