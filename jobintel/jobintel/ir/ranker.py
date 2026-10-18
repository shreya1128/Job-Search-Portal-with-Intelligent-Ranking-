"""Ranking signals, normalisation and the configurable hybrid IR score.

    IR score = ( w1*LM + w2*DFR + w3*Passage + w4*SkillMatch + w5*Location + w6*Experience + w7*PageRank ) / sum(active weights)

Every signal is normalised to [0, 1] BEFORE combining (min-max over the candidate pool for LM / DFR / passage / PageRank; the others are
already in [0, 1]).  A signal that does not apply to a query (no location typed, no skills in the query ...) is None and its weight is
dropped, so the remaining weights are re-normalised.  The default weights are a starting point, NOT claimed optimal - Phase 4 evaluates
alternatives."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# PROVISIONAL defaults (sum = 1): a city typed in a natural query is user INTENT, so location counts for much more than a tie-breaker.
# Not tuned and not claimed optimal - Phase 4 compares alternatives with relevance judgments.
DEFAULT_WEIGHTS: Dict[str, float] = {"lm": 0.25, "dfr": 0.15, "passage": 0.12, "skill": 0.12, "location": 0.20, "experience": 0.08, "pagerank": 0.08}
UNKNOWN_SCORE = 0.3          # neutral-but-below-average value when the document has no data for a signal (e.g. no experience range)


@dataclass
class RankerConfig:
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    lm_smoothing: str = "dirichlet"
    mu: float = 200.0
    lam: float = 0.3
    dfr_model: str = "PL2"
    dfr_c: float = 1.0
    bigram_mu_b: float = 100.0
    bigram_alpha: float = 0.5
    pool: int = 100
    passage_mu: float = 50.0


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return x
    finite = np.isfinite(x)
    if not finite.any():
        return np.full(len(x), 0.5)
    lo, hi = x[finite].min(), x[finite].max()
    out = np.zeros(len(x))
    if hi - lo < 1e-12:
        out[finite] = 0.5                                    # no discrimination between candidates
        return out
    out[finite] = (x[finite] - lo) / (hi - lo)
    return out


def location_score(doc_cities: Optional[Sequence[str]], doc_city: Optional[str], wanted: Sequence[str]) -> float:
    have = {c.lower() for c in (doc_cities or ([doc_city] if doc_city else []))}
    if not have:
        return UNKNOWN_SCORE
    return 1.0 if have & {w.lower() for w in wanted} else 0.0


def experience_score(doc_min: float, doc_max: float, q_min: float, q_max: Optional[float]) -> float:
    """1.0 when the posting's experience band overlaps the wanted band; otherwise 1/(1+gap in years)."""
    if math.isnan(doc_min):
        return UNKNOWN_SCORE
    d_hi = math.inf if math.isnan(doc_max) else doc_max
    q_hi = math.inf if q_max is None else q_max
    if doc_min <= q_hi and q_min <= d_hi:
        return 1.0
    gap = (doc_min - q_hi) if doc_min > q_hi else (q_min - d_hi)
    return 1.0 / (1.0 + gap)


def skill_score(doc_skills: Set[str], wanted: Sequence[str]) -> Tuple[float, List[str], List[str]]:
    matched = [s for s in wanted if s in doc_skills]
    missing = [s for s in wanted if s not in doc_skills]
    return (len(matched) / len(wanted) if wanted else 0.0), matched, missing


def combine(signals: Dict[str, Optional[np.ndarray]], weights: Dict[str, float], n: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """Weighted mean over the signals that apply.  Returns (final score per candidate, effective normalised weights)."""
    active = {k: weights.get(k, 0.0) for k, v in signals.items() if v is not None and weights.get(k, 0.0) > 0}
    total = sum(active.values())
    if n is None:
        n = len(next(v for v in signals.values() if v is not None)) if active else 0
    if not active or total <= 0:
        return np.zeros(n), {}
    eff = {k: w / total for k, w in active.items()}
    final = sum(eff[k] * np.asarray(signals[k], dtype=np.float64) for k in eff)
    return final, eff
