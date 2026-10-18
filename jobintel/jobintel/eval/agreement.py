"""Inter-rater agreement between two sets of relevance labels (e.g. a HUMAN's and the AI judge's) on the same jobs.

  cohen_kappa(a, b)                unweighted Cohen's kappa over the labels 0 / 1 / 2
  cohen_kappa(a, b, 'quadratic')   weighted kappa: a 0-vs-2 disagreement costs 4x a 0-vs-1 disagreement (appropriate for ordered grades)
  agreement_report(pairs)          everything above + confusion matrix, binary agreement at thresholds 1 and 2, per-query agreement and a bootstrap interval over QUERIES
Interpretation caution: kappa depends on the label distribution (most pooled jobs are relevant, which depresses kappa for a given raw agreement) and the sample is small;
the bootstrap interval resamples whole queries because judgments within a query are not independent."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LABELS = (0, 1, 2)


def confusion(a: Sequence[int], b: Sequence[int], labels: Sequence[int] = LABELS) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for x, y in zip(a, b):
        m[idx[int(x)], idx[int(y)]] += 1
    return m


def cohen_kappa(a: Sequence[int], b: Sequence[int], weights: Optional[str] = None, labels: Sequence[int] = LABELS) -> float:
    if weights not in (None, "linear", "quadratic"):
        raise ValueError("weights must be None, 'linear' or 'quadratic'")
    O = confusion(a, b, labels).astype(float)
    n = O.sum()
    if n == 0:
        return float("nan")
    O /= n
    E = np.outer(O.sum(axis=1), O.sum(axis=0))
    k = len(labels)
    i, j = np.indices((k, k))
    W = np.abs(i - j).astype(float) if weights == "linear" else ((i - j) ** 2).astype(float) if weights == "quadratic" else (i != j).astype(float)
    den = (W * E).sum()
    return float("nan") if den == 0 else float(1.0 - (W * O).sum() / den)


def _binary(a, b, thr):
    return [int(x >= thr) for x in a], [int(y >= thr) for y in b]


def agreement_report(pairs: Sequence[Tuple[str, int, int]], n_boot: int = 2000, seed: int = 0) -> Dict:
    """pairs = [(query_id, label_a, label_b)] for jobs labelled by BOTH raters."""
    q = np.array([p[0] for p in pairs])
    a = np.array([p[1] for p in pairs], dtype=int)
    b = np.array([p[2] for p in pairs], dtype=int)
    out: Dict = {"n_pairs": int(len(pairs)), "n_queries": int(len(set(q.tolist())))}
    if not len(pairs):
        return out
    out["exact_agreement"] = float((a == b).mean())
    out["within_one_grade"] = float((np.abs(a - b) <= 1).mean())
    out["kappa"] = cohen_kappa(a, b)
    out["weighted_kappa_quadratic"] = cohen_kappa(a, b, "quadratic")
    out["confusion_rows_a_cols_b"] = confusion(a, b).tolist()
    out["mean_label"] = {"a": float(a.mean()), "b": float(b.mean())}
    for thr in (1, 2):
        x, y = _binary(a, b, thr)
        out[f"binary_thr{thr}"] = {"agreement": float(np.mean(np.array(x) == np.array(y))), "kappa": cohen_kappa(x, y, labels=(0, 1)), "share_relevant_a": float(np.mean(x)), "share_relevant_b": float(np.mean(y))}
    qs = sorted(set(q.tolist()))
    out["per_query"] = {qq: {"n": int((q == qq).sum()), "exact": float((a[q == qq] == b[q == qq]).mean()), "mean_a": float(a[q == qq].mean()), "mean_b": float(b[q == qq].mean())} for qq in qs}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(qs, size=len(qs), replace=True)
        idx = np.concatenate([np.where(q == p)[0] for p in pick])
        k = cohen_kappa(a[idx], b[idx], "quadratic")
        if k == k:
            boots.append(k)
    if len(boots) > 20:
        out["weighted_kappa_ci95_over_queries"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out
