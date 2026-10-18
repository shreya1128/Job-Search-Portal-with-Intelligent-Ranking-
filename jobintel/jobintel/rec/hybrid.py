"""The configurable 11-signal hybrid recommender.

    final(user, job) = sum_s w_s * x_s(user, job) / sum of the ACTIVE weights          x_s in [0, 1]

  ir           the Phase-2 hybrid IR score of the job for a query built from the profile (role + top skills + city)      lm / dfr / passage / pagerank: its components
  dl           Phase 5 embedding similarity (ADAPTER - inactive until real embeddings exist, see dlsim.py)
  content      idf-weighted skill-vector cosine            skill_match  share of the job's skill requirements the profile covers
  knowledge    knowledge-graph occupation coverage (cold-start capable)      collaborative  item-based CF from the user's interactions (None for a new user)
  preference   mean of the preference checks that apply (preferred city, experience band, category, employment type)
A signal that cannot be computed for this user (no history, no embeddings, no preferences ...) is None; its weight is dropped and the others are re-normalised
(`ranker.combine`), and the response lists WHY each inactive signal is inactive.  The default weights are PROVISIONAL starting values - not tuned, not claimed optimal.
`ir` already blends lm / dfr / passage / pagerank; keeping those as separate (small) weights lets the user emphasise a component, at the price of some double counting.
Jobs that the IR query did not retrieve get 0 for the IR-family signals ('not retrieved'); content / knowledge / collaborative / dl are min-max normalised over the candidate pool."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from ..ir.ranker import UNKNOWN_SCORE, combine, experience_score, location_score, minmax
from .catalog import ItemCatalog
from .dlsim import DLSimilarity
from .interactions import ItemKNN
from .knowledge import KnowledgeGraph
from .profile import UserProfile

SIGNALS = ["ir", "lm", "dfr", "passage", "pagerank", "dl", "content", "collaborative", "knowledge", "skill_match", "preference"]
DEFAULT_REC_WEIGHTS: Dict[str, float] = {"ir": 0.15, "lm": 0.05, "dfr": 0.05, "passage": 0.05, "pagerank": 0.04, "dl": 0.10, "content": 0.15, "collaborative": 0.10,
                                         "knowledge": 0.10, "skill_match": 0.12, "preference": 0.09}
NORMALISE = {"content", "collaborative", "knowledge", "dl"}


@dataclass
class Recommendation:
    rank: int
    job_id: int
    title: str
    company: Optional[str]
    city: Optional[str]
    source: str
    final_score: float
    signals: Dict[str, Optional[float]]
    contributions: Dict[str, float]
    matched_skills: List[str]
    missing_skills: List[str]
    reasons: List[str]
    cities: List[str] = field(default_factory=list)


@dataclass
class RecommendationResponse:
    results: List[Recommendation] = field(default_factory=list)
    weights_requested: Dict[str, float] = field(default_factory=dict)
    weights_effective: Dict[str, float] = field(default_factory=dict)
    inactive: Dict[str, str] = field(default_factory=dict)
    ir_query: Optional[str] = None
    n_candidates: int = 0
    dl_coverage: Optional[float] = None                      # share of the candidates that have a DL embedding (None: DL inactive)
    knowledge_info: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def preference_score(cat: ItemCatalog, di: int, p: UserProfile) -> Optional[float]:
    m, parts = cat.ix.meta, []
    if p.locations:
        parts.append(location_score(m["cities"][di], m["city"][di], p.locations))
    if p.experience_years is not None:
        parts.append(experience_score(m["exp_min"][di], m["exp_max"][di], max(0.0, p.experience_years - 1.0), p.experience_years + 1.0))
    if p.categories:
        c = (m["category"][di] or "").lower()
        parts.append(UNKNOWN_SCORE if not c else float(c in {x.lower() for x in p.categories}))
    if p.employment:
        e = (m["employment"][di] or "").lower()
        parts.append(UNKNOWN_SCORE if not e else float(e == p.employment.lower()))
    return float(np.mean(parts)) if parts else None


class HybridRecommender:
    def __init__(self, engine, catalog: Optional[ItemCatalog] = None, kg: Optional[KnowledgeGraph] = None, cf: Optional[ItemKNN] = None,
                 dl=None,  # DLSimilarity | DLSkillSimilarity | None — both expose .available / .reason / .score_docs
                 weights: Optional[Dict[str, float]] = None):
        self.engine, self.cat = engine, catalog or ItemCatalog(engine.index)
        self.kg, self.cf, self.dl = kg, cf, dl if dl is not None else DLSimilarity()
        self.weights = {**DEFAULT_REC_WEIGHTS, **(weights or {})}
        unknown = set(self.weights) - set(SIGNALS)
        if unknown:
            raise ValueError(f"unknown signal(s): {sorted(unknown)}")


    # -------------------------------------------------------------------------------------------------------- helpers
    def _ir_query(self, p: UserProfile) -> Optional[str]:
        parts = ([p.preferred_role] if p.preferred_role else []) + sorted(p.skills, key=lambda s: -p.skills[s])[:6] + p.locations[:1]
        q = " ".join(x for x in parts if x).strip()
        return q[:190] or None

    @staticmethod
    def _top(arr: Optional[np.ndarray], n: int, pool: Set[int]) -> None:
        if arr is not None and n > 0:
            idx = np.argpartition(-arr, min(n, len(arr) - 1))[:n]
            pool.update(int(i) for i in idx if arr[i] > 0)

    # -------------------------------------------------------------------------------------------------------- main
    def recommend(self, profile: UserProfile, k: int = 10, *, pool_n: int = 100, weights: Optional[Dict[str, float]] = None, exclude_job_ids: Optional[Sequence[int]] = None,
                  collapse_duplicates: bool = True, strict_location: bool = False, restrict_docs: Optional[Sequence[int]] = None) -> RecommendationResponse:
        W = {**self.weights, **(weights or {})}
        resp = RecommendationResponse(weights_requested=dict(W))
        cat, ix = self.cat, self.cat.ix
        inactive: Dict[str, str] = {}
        mask = None                                              # strict_location: only jobs in the preferred cities are candidates (default: location is ONE soft signal of 11)
        if strict_location and profile.locations:
            parts = [p_ for p_ in (ix.city_docs.get(c.lower()) for c in profile.locations) if p_ is not None]
            if not parts:
                resp.warnings.append(f"strict_location: no job in {profile.locations}")
                resp.inactive = {s_: "no candidates" for s_ in SIGNALS}
                return resp
            mask = np.zeros(cat.N, dtype=bool)
            mask[np.concatenate(parts)] = True
        # ---- full-catalogue signals ------------------------------------------------------------------------------
        known = {s: w for s, w in profile.skills.items() if s in cat.sidx}
        unknown = sorted(set(profile.skills) - set(known))
        if unknown:
            resp.warnings.append(f"skills not present in any job of the catalogue (ignored by the skill-space signals): {unknown[:8]}")
        content = cat.content_scores(known) if known else None
        coverage = cat.coverage(known) if known else None
        if content is None:
            inactive["content"] = inactive["skill_match"] = "the profile has no skill that occurs in the job catalogue"
        knowledge = None
        if self.kg is not None:
            knowledge, info = self.kg.knowledge_scores(cat, profile.skills, profile.preferred_role)
            resp.knowledge_info = info
            if knowledge is None:
                inactive["knowledge"] = info.get("reason", "no knowledge-graph evidence")
        else:
            inactive["knowledge"] = "no knowledge graph loaded"
        collab = None
        hist = {ix.doc_index(j): w for j, w in profile.interactions.items() if ix.doc_index(j) is not None}
        if self.cf is None:
            inactive["collaborative"] = "no collaborative model fitted"
        elif not hist:
            inactive["collaborative"] = "no interaction history for this user (cold start)"
        else:
            collab = self.cf.score(hist)
            if collab is None:
                inactive["collaborative"] = "none of the user's jobs is known to the collaborative model"
        # ---- IR candidates ------------------------------------------------------------------------------------------
        q = self._ir_query(profile)
        resp.ir_query = q
        ir_rows: Dict[int, Dict] = {}
        if q is None:
            inactive.update({s: "no role or skills to build an IR query from" for s in ("ir", "lm", "dfr", "passage", "pagerank")})
        else:
            r = self.engine.search(q, k=pool_n, model="hybrid", explain=False, filters={"city": profile.locations} if mask is not None else None)
            if r.error or not r.results:
                why = r.error or "the IR query matched no job"
                inactive.update({s: why for s in ("ir", "lm", "dfr", "passage", "pagerank")})
            else:
                for x in r.results:
                    di = ix.doc_index(x.job_id)
                    if di is not None:
                        ir_rows[di] = {"ir": x.score, **{s: x.signals.get(s) for s in ("lm", "dfr", "passage", "pagerank")}}
                for s in ("lm", "dfr", "passage", "pagerank"):
                    if all(v[s] is None for v in ir_rows.values()):
                        inactive[s] = "not produced by the IR run" + (" (the index has no PageRank)" if s == "pagerank" else "")
        # ---- candidate pool -----------------------------------------------------------------------------------------
        pool: Set[int] = set(ir_rows)
        for arr in (content, knowledge, collab, coverage if known else None):
            self._top(arr if (mask is None or arr is None) else np.where(mask, arr, 0.0), pool_n, pool)
        if mask is not None:
            pool = {d for d in pool if mask[d]}
        excl = {ix.doc_index(j) for j in (exclude_job_ids or [])} - {None}
        if restrict_docs is not None:                             # evaluation mode: rank exactly this candidate set (signals are still computed the normal way)
            pool = {int(d) for d in restrict_docs}
        pool -= excl
        cand = np.array(sorted(pool), dtype=np.int64)
        resp.n_candidates = int(len(cand))
        if not len(cand):
            resp.inactive = inactive
            resp.warnings.append("no candidate jobs: give a role and / or skills")
            return resp
        # ---- signal matrix over the candidates ----------------------------------------------------------------------
        sig: Dict[str, Optional[np.ndarray]] = {s: None for s in SIGNALS}
        if ir_rows:
            for s in ("ir", "lm", "dfr", "passage", "pagerank"):
                if s not in inactive:
                    sig[s] = np.array([float(ir_rows.get(int(d), {}).get(s) or 0.0) for d in cand])
        if self.dl.available:
            dl = self.dl.score_docs(profile, cat, cand)
            fin = None if dl is None else np.isfinite(dl)
            if dl is None or not fin.any():
                inactive["dl"] = "the profile has no skill in the DL vocabulary, or no candidate has an embedding"
            else:
                resp.dl_coverage = float(fin.mean())
                v = np.full(len(dl), UNKNOWN_SCORE)                    # a job WITHOUT an embedding is neutral: neither penalised nor boosted
                v[fin] = minmax(dl[fin])
                sig["dl"] = v
        else:
            inactive["dl"] = self.dl.reason or "DL similarity not available"
        for name, arr in (("content", content), ("collaborative", collab), ("knowledge", knowledge)):
            if arr is not None:
                sig[name] = minmax(arr[cand])
        if coverage is not None:
            sig["skill_match"] = coverage[cand]
        pref = [preference_score(cat, int(d), profile) for d in cand]
        if all(v is None for v in pref):
            inactive["preference"] = "no preferences given (city / experience / category / employment)"
        else:
            sig["preference"] = np.array([UNKNOWN_SCORE if v is None else v for v in pref])
        sig = {k_: (None if v is None else np.asarray(v, dtype=np.float64)) for k_, v in sig.items()}      # one precision everywhere: contributions must sum to the score
        final, eff = combine(sig, W, len(cand))
        resp.weights_effective = dict(eff)                      # exact (unrounded): the response must reproduce its own scores
        resp.inactive = {s: inactive.get(s, "weight is 0" if W.get(s, 0) <= 0 else "inactive") for s in SIGNALS if s not in eff}
        order = np.lexsort((cand, -final))
        # ---- results ----------------------------------------------------------------------------------------------
        seen, rank = set(), 0
        have = set(profile.skills)
        for o in order:
            d = int(cand[o])
            m = ix.meta
            if collapse_duplicates:
                key = (str(m["title"][d]).lower(), str(m["company"][d] or "").lower(), tuple(m["cities"][d] or [m["city"][d]]))
                if key in seen:
                    continue
                seen.add(key)
            jskills = cat.job_skills(d)
            matched = [s for s in jskills if s in have]
            missing = [s for s in jskills if s not in have]
            contrib = {s: float(eff[s] * sig[s][o]) for s in eff}
            top3 = sorted(contrib, key=lambda s: -contrib[s])[:3]
            reasons = [f"strongest signals: {', '.join(top3)}"]
            if matched:
                reasons.append(f"you have {len(matched)} of its {len(jskills)} listed skills: {', '.join(matched[:6])}")
            if missing:
                reasons.append(f"missing: {', '.join(missing[:6])}")
            jcities = list(m["cities"][d] or ([m["city"][d]] if m["city"][d] else []))
            hit_city = [c for c in jcities if c.lower() in {x.lower() for x in profile.locations}]
            if profile.locations and sig["preference"] is not None and hit_city:
                reasons.append(f"lists your preferred city ({hit_city[0]})" + (f" among {len(jcities)} cities" if len(jcities) > 1 else ""))
            rank += 1
            resp.results.append(Recommendation(rank, int(ix.job_ids[d]), m["title"][d], m["company"][d], m["city"][d], m["source"][d], float(final[o]),
                                               {s: (None if sig[s] is None else float(sig[s][o])) for s in SIGNALS}, contrib, matched, missing, reasons, jcities))
            if rank >= k:
                break
        if profile.locations and resp.results and mask is None:
            wanted = {c.lower() for c in profile.locations}
            def in_city(r_):
                d_ = ix.doc_index(r_.job_id)
                return bool(wanted & {c.lower() for c in (ix.meta["cities"][d_] or [ix.meta["city"][d_]] if ix.meta["city"][d_] else [])})
            if not any(in_city(r_) for r_ in resp.results):
                resp.warnings.append(f"none of the top {len(resp.results)} jobs is in {profile.locations}: location is only one of the weighted signals here - use strict_location=True to require it")
        return resp
