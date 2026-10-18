"""Search engine: query processing -> candidate retrieval -> ranking model -> (re-rank with passages & signals) -> explained results.

    USER QUERY -> QUERY PROCESSOR -> [filters] -> CANDIDATES (docs containing >= 1 query term)
        baseline | lm | lm_bigram | dfr   rank all candidates by that single model
        passage                            best-passage language-model score on the LM shortlist
        hybrid                             shortlist = top-N by LM  UNION  top-N by DFR, then re-rank the shortlist with
                                           LM + DFR + passage + skill + location + experience + PageRank  (configurable weights)
Every result carries the reasons it matched (terms, skills, location, best passage with highlighting, scores)."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from jobintel.extraction.extractor import Extractor
from jobintel.extraction.lexicon import SkillLexicon, load_lexicon
from jobintel.ir import models as M
from jobintel.ir import passages as PS
from jobintel.ir.index import InvertedIndex
from jobintel.ir.query import ParsedQuery, QueryProcessor
from jobintel.ir.ranker import RankerConfig, combine, experience_score, location_score, minmax, skill_score

MODELS = ("baseline", "lm", "lm_bigram", "dfr", "passage", "hybrid")
_EMPTY = np.zeros(0, dtype=np.int32)


# =========================================================================== document stores (original text for passages / display)
class MemoryDocStore:
    def __init__(self, docs: Dict[int, Tuple[str, str]]):
        self.docs = docs

    def get(self, job_ids: Sequence[int]) -> Dict[int, Tuple[str, str]]:
        return {j: self.docs[j] for j in job_ids if j in self.docs}


class DbDocStore:
    """Fetches title + display text of the few documents that need passages (only the shortlist, never the whole collection)."""

    def __init__(self, engine):
        self.engine = engine

    def get(self, job_ids: Sequence[int]) -> Dict[int, Tuple[str, str]]:
        from sqlalchemy import select
        from jobintel.db.models import Job
        out: Dict[int, Tuple[str, str]] = {}
        ids = [int(j) for j in job_ids]
        with self.engine.connect() as c:
            for i in range(0, len(ids), 500):
                for jid, title, text in c.execute(select(Job.job_id, Job.title, Job.description_clean).where(Job.job_id.in_(ids[i:i + 500]))):
                    out[jid] = (title or "", text or "")
        return out


# =========================================================================== results
@dataclass
class SearchResult:
    rank: int
    job_id: int
    title: str
    company: Optional[str]
    city: Optional[str]
    cities: List[str]
    country: Optional[str]
    source: str
    score: float
    signals: Dict[str, Optional[float]]
    raw: Dict[str, float]
    passage: Optional[Dict]
    reasons: List[str]
    experience: Optional[str]
    salary: Optional[str]
    skills: List[str]
    text_quality: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SearchResponse:
    query: str
    model: str
    parsed: Optional[ParsedQuery] = None
    results: List[SearchResult] = field(default_factory=list)
    n_candidates: int = 0
    n_docs: int = 0
    weights: Dict[str, float] = field(default_factory=dict)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    routing: Optional[Dict] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["parsed"] = self.parsed.to_dict() if self.parsed else None
        return d


def _fmt_exp(lo: float, hi: float) -> Optional[str]:
    if math.isnan(lo):
        return None
    if math.isnan(hi):
        return f"{lo:g}+ yrs"
    return f"{lo:g} yrs" if lo == hi else f"{lo:g}-{hi:g} yrs"


def _fmt_salary(lo, hi, cur, per) -> Optional[str]:
    if math.isnan(lo):
        return None
    rng = f"{lo:,.0f}" if lo == hi or math.isnan(hi) else f"{lo:,.0f}-{hi:,.0f}"
    return f"{cur or ''} {rng}{' / ' + per if per else ''}".strip()


# =========================================================================== first-stage result shared by the local and the distributed engine
@dataclass
class Stage:
    """What the first (retrieval) stage hands to the second (re-ranking / explanation) stage, as GLOBAL document indices.
    ranked : documents in the model's order (single-model rankings)      pool : shortlist for passage / hybrid / browse
    lm, dfr, score : raw scores of the documents that were returned       extra : timings reported by a distributed coordinator"""
    n_candidates: int
    ranked: np.ndarray
    pool: np.ndarray
    lm: Dict[int, float] = field(default_factory=dict)
    dfr: Dict[int, float] = field(default_factory=dict)
    score: Dict[int, float] = field(default_factory=dict)
    extra: Dict[str, float] = field(default_factory=dict)


# =========================================================================== engine
class SearchEngine:
    def __init__(self, index: InvertedIndex, docstore, extractor: Optional[Extractor] = None, config: Optional[RankerConfig] = None,
                 lexicon: Optional[SkillLexicon] = None):
        self.index, self.docstore = index, docstore
        self.lex = lexicon or load_lexicon()
        self.extractor = extractor or Extractor(self.lex)
        self.processor = QueryProcessor(self.extractor)
        self.config = config or RankerConfig()
        self.router = None

    def _router(self):
        if self.router is None:
            from jobintel.ir.cluster import ClusterRouter
            self.router = ClusterRouter.from_index(self.index)
        return self.router

    # ------------------------------------------------------------------ filters (hard constraints)
    def _allowed(self, f: Dict) -> Optional[np.ndarray]:
        ix = self.index
        sel: Optional[np.ndarray] = None

        def narrow(arr: np.ndarray) -> None:
            nonlocal sel
            sel = np.unique(arr) if sel is None else np.intersect1d(sel, arr, assume_unique=False)

        as_list = lambda v: [v] if isinstance(v, str) else list(v)
        loc = f.get("city") or f.get("location")
        if loc:
            narrow(np.concatenate([ix.city_docs.get(c.lower(), _EMPTY) for c in as_list(loc)] or [_EMPTY]))
        for s in as_list(f.get("skills") or []):
            narrow(ix.skill_docs.get(self.lex.normalize(s) or s.lower(), _EMPTY))
        preds = {"country", "sources", "cluster", "category", "role", "employment_type", "min_experience", "max_experience", "min_salary", "text_quality"}
        if not preds & {k for k, v in f.items() if v not in (None, "", [])}:
            return sel
        base = sel if sel is not None else np.arange(ix.n_docs)
        m = np.ones(len(base), dtype=bool)
        if f.get("country"):
            m &= np.isin(ix.country_lc[base], [c.lower() for c in as_list(f["country"])])
        if f.get("sources"):
            m &= np.isin(ix.source_arr[base], as_list(f["sources"]))
        if f.get("cluster"):
            m &= np.isin(ix.cluster_arr[base], as_list(f["cluster"]))
        if f.get("text_quality"):
            m &= np.isin(ix.quality_arr[base], as_list(f["text_quality"]))
        if f.get("category"):
            wanted = [c.lower() for c in as_list(f["category"])]
            m &= np.array([any(w in c for w in wanted) for c in ix.category_lc[base]], dtype=bool)
        if f.get("role"):
            wanted = [r.lower() for r in as_list(f["role"])]
            m &= np.array([any(w in t for w in wanted) for t in ix.title_lc[base]], dtype=bool)
        if f.get("employment_type"):
            wanted = [r.lower() for r in as_list(f["employment_type"])]
            m &= np.array([any(w in t for w in wanted) for t in ix.employment_lc[base]], dtype=bool)
        if f.get("min_experience") is not None or f.get("max_experience") is not None:
            lo, hi = f.get("min_experience"), f.get("max_experience")
            emin, emax = ix.meta["exp_min"][base], ix.meta["exp_max"][base]
            ok = ~np.isnan(emin)                                              # unknown experience cannot satisfy an explicit experience filter
            if hi is not None:
                ok &= emin <= hi
            if lo is not None:
                ok &= np.isnan(emax) | (emax >= lo)
            m &= ok
        if f.get("min_salary") is not None:
            smax = ix.meta["sal_max"][base]
            m &= ~np.isnan(smax) & (smax >= f["min_salary"])
        return base[m]

    @staticmethod
    def _top(raw: np.ndarray, n: int) -> np.ndarray:
        return M.top_positions(raw, n)

    # ------------------------------------------------------------------ first stage (local implementation)
    def _stage(self, model: str, parsed: ParsedQuery, qterms, allowed: Optional[np.ndarray], fetch_k: int, browse: bool) -> Stage:
        ix, cfg = self.index, self.config
        E = np.zeros(0, dtype=np.int64)
        if browse:
            cands = allowed if allowed is not None else np.arange(ix.n_docs)
            if parsed.locations:                                  # nothing to match as text: the typed city IS the request -> restrict to it
                in_loc = np.unique(np.concatenate([ix.city_docs.get(l.lower(), _EMPTY) for l in parsed.locations] or [_EMPTY]))
                cands = np.intersect1d(cands, in_loc)
            prior = ix.pagerank[cands] if ix.pagerank is not None else np.zeros(len(cands))
            pool = cands[self._top(prior, min(len(cands), 1000))]
            return Stage(int(len(cands)), pool, pool)
        cands = M.candidates(ix, qterms)
        if allowed is not None:
            cands = np.intersect1d(cands, allowed, assume_unique=True)
        n = int(len(cands))
        if not n:
            return Stage(0, E, E)
        smap = lambda raw, pos: {int(cands[p]): float(raw[p]) for p in pos}
        if model == "baseline":
            raw = M.keyword_baseline(ix, qterms, cands)
            pos = self._top(raw, fetch_k)
            return Stage(n, cands[pos], cands[pos], score=smap(raw, pos))
        if model == "lm":
            raw = M.unigram_lm(ix, qterms, cands, smoothing=cfg.lm_smoothing, mu=cfg.mu, lam=cfg.lam)
            pos = self._top(raw, fetch_k)
            return Stage(n, cands[pos], cands[pos], lm=smap(raw, pos), score=smap(raw, pos))
        if model == "lm_bigram":
            raw = M.bigram_lm(ix, parsed.sequence(ix), cands, mu=cfg.mu, mu_b=cfg.bigram_mu_b, alpha=cfg.bigram_alpha)
            pos = self._top(raw, fetch_k)
            return Stage(n, cands[pos], cands[pos], lm=smap(raw, pos), score=smap(raw, pos))
        if model == "dfr":
            raw = M.dfr(ix, qterms, cands, model=cfg.dfr_model, c=cfg.dfr_c)
            pos = self._top(raw, fetch_k)
            return Stage(n, cands[pos], cands[pos], dfr=smap(raw, pos), score=smap(raw, pos))
        lm_raw = M.unigram_lm(ix, qterms, cands, smoothing=cfg.lm_smoothing, mu=cfg.mu, lam=cfg.lam)      # passage | hybrid -> shortlist
        pool_pos = self._top(lm_raw, cfg.pool)
        dfr_raw = None
        if model == "hybrid":
            dfr_raw = M.dfr(ix, qterms, cands, model=cfg.dfr_model, c=cfg.dfr_c)
            pool_pos = np.unique(np.concatenate([pool_pos, self._top(dfr_raw, cfg.pool)]))
        else:
            pool_pos = np.sort(pool_pos)                          # shortlist in DOCUMENT order (same convention as the distributed merger)
        pool = cands[pool_pos]
        return Stage(n, E, pool, lm=smap(lm_raw, pool_pos), dfr=smap(dfr_raw, pool_pos) if dfr_raw is not None else {})

    def _passages(self, docs: np.ndarray, qterms) -> Dict[int, Tuple[Optional[PS.Passage], float, str]]:
        ix = self.index
        texts = self.docstore.get([int(ix.job_ids[d]) for d in docs])
        info = [(ix.terms[t], n, M.collection_prob(ix, t)) for t, n in qterms]
        out = {}
        for d in docs:
            title, text = texts.get(int(ix.job_ids[d]), ("", ""))
            ps = PS.segment(text)
            sc = PS.score_passages(ps, info, mu=self.config.passage_mu)
            bp, bs = PS.best_passage(ps, sc)
            out[int(d)] = (bp, bs, text)
        return out

    # ------------------------------------------------------------------ side-by-side comparison of the ranking models
    def compare_models(self, query: str, k: int = 10, models: Sequence[str] = ("baseline", "lm", "lm_bigram", "dfr", "hybrid"), **kw) -> Dict:
        """For one query: the rank every model gives to every job that any model put in its top k, plus the normalised LM and DFR scores.
        Different models rank differently - this is the evidence."""
        runs = {m: self.search(query, k=k, model=m, explain=False, **kw) for m in models}
        rows: Dict[int, Dict] = {}
        for m, r in runs.items():
            for x in r.results:
                row = rows.setdefault(x.job_id, {"job_id": x.job_id, "title": x.title, "company": x.company, "ranks": {}})
                row["ranks"][m] = x.rank
        ix = self.index
        qterms, _ = runs[models[0]].parsed.qterms(ix) if runs[models[0]].parsed else ([], [])
        if rows and qterms:
            docs = np.array([ix.doc_index(j) for j in rows])
            lm = M.unigram_lm(ix, qterms, docs, smoothing=self.config.lm_smoothing, mu=self.config.mu, lam=self.config.lam) if False else None
            cand_all = M.candidates(ix, qterms)
            lm_all = M.unigram_lm(ix, qterms, cand_all, smoothing=self.config.lm_smoothing, mu=self.config.mu, lam=self.config.lam)
            dfr_all = M.dfr(ix, qterms, cand_all, model=self.config.dfr_model, c=self.config.dfr_c)
            lm_n, dfr_n = minmax(lm_all), minmax(dfr_all)
            for j, row in rows.items():
                pos = int(np.searchsorted(cand_all, ix.doc_index(j)))
                row["lm_score"], row["dfr_score"] = round(float(lm_n[pos]), 3), round(float(dfr_n[pos]), 3)
        ordered = sorted(rows.values(), key=lambda r: (r["ranks"].get("hybrid", 10**6), min(r["ranks"].values())))
        return {"query": query, "models": list(models), "rows": ordered, "timings_ms": {m: r.timings_ms.get("total") for m, r in runs.items()}}

    # ------------------------------------------------------------------ main entry point
    def search(self, query: str, *, k: int = 10, model: str = "hybrid", filters: Optional[Dict] = None, weights: Optional[Dict[str, float]] = None,
               explain: bool = True, collapse_duplicates: bool = False, cluster_routing: bool = False) -> SearchResponse:
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        t0 = time.perf_counter()
        cfg, ix = self.config, self.index
        w = {**cfg.weights, **(weights or {})}
        resp = SearchResponse(query=query, model=model, n_docs=ix.n_docs)
        parsed = self.processor.parse(query)
        resp.parsed = parsed
        if not parsed.valid:
            resp.error = parsed.error
            return resp
        qterms, unknown = parsed.qterms(ix)
        if unknown:
            resp.warnings.append(f"terms not found in the index (ignored): {unknown}")
        allowed = self._allowed(filters or {})
        if cluster_routing and parsed.terms and qterms and not (filters or {}).get("cluster"):
            router = self._router()
            route = router.route(qterms)
            in_clusters = router.docs_in(route.clusters)
            allowed = in_clusters if allowed is None else np.intersect1d(allowed, in_clusters)
            resp.routing = route.to_dict()
        t1 = time.perf_counter()
        if parsed.terms and not qterms:                       # the user typed words, but none of them exist in the index
            resp.warnings.append("none of the query terms occur in the index")
            resp.timings_ms = {"total": round((time.perf_counter() - t0) * 1000, 1)}
            return resp
        browse = not parsed.terms                             # browse mode only for queries like 'jobs in Pune' (nothing left to match as text)
        if browse and model != "hybrid":
            resp.warnings.append("query has no searchable words after removing locations/levels/generic words")
            return resp
        fetch_k = k * 3 if collapse_duplicates else k         # over-fetch so that collapsing repeated postings still leaves k results
        stage = self._stage(model, parsed, qterms, allowed, fetch_k, browse)
        resp.n_candidates = stage.n_candidates
        if browse:
            resp.warnings.append("no text terms in the query: showing filtered jobs ranked by the non-text signals")
        if stage.n_candidates == 0 or (browse and not len(stage.pool)):
            resp.timings_ms = {"total": round((time.perf_counter() - t0) * 1000, 1), **stage.extra}
            return resp
        t2 = time.perf_counter()
        total_q = sum(n for _, n in qterms) or 1

        passage_info: Dict[int, Tuple[Optional[PS.Passage], float, str]] = {}
        t_pass = 0.0
        raw_final: Optional[Dict[int, float]] = None
        sig_at: Dict[int, Dict] = {}
        if browse or model in ("hybrid", "passage"):
            pool_docs = stage.pool
            n = len(pool_docs)
            if not browse:
                tp = time.perf_counter()
                passage_info = self._passages(pool_docs, qterms)
                t_pass = time.perf_counter() - tp
            sig: Dict[str, Optional[np.ndarray]] = {"lm": None, "dfr": None, "passage": None, "skill": None, "location": None, "experience": None, "pagerank": None}
            if not browse:
                if model == "hybrid":
                    sig["lm"] = minmax(np.array([stage.lm[int(d)] for d in pool_docs]) / total_q)
                    sig["dfr"] = minmax(np.array([stage.dfr[int(d)] for d in pool_docs]))
                sig["passage"] = minmax(np.array([passage_info[int(d)][1] for d in pool_docs]))
            if model != "passage":
                if parsed.skills:
                    sig["skill"] = np.array([skill_score(set(ix.meta["skills"][d] or []), parsed.skills)[0] for d in pool_docs])
                if parsed.locations:
                    sig["location"] = np.array([location_score(ix.meta["cities"][d], ix.meta["city"][d], parsed.locations) for d in pool_docs])
                if parsed.experience:
                    e = parsed.experience
                    sig["experience"] = np.array([experience_score(ix.meta["exp_min"][d], ix.meta["exp_max"][d], e["min"], e["max"]) for d in pool_docs])
                if ix.pagerank is not None:
                    sig["pagerank"] = minmax(np.log(ix.pagerank[pool_docs] + 1e-12))
            if model == "passage":
                final, resp.weights = sig["passage"], {"passage": 1.0}
            else:
                final, resp.weights = combine(sig, w, n)
            order_in_pool = np.lexsort((np.arange(n), -final))[:fetch_k]
            order_docs = pool_docs[order_in_pool]
            raw_final = {int(pool_docs[i]): float(final[i]) for i in order_in_pool}
            sig_at = {int(pool_docs[i]): {s: (None if v is None else float(v[i])) for s, v in sig.items()} for i in order_in_pool}
        else:
            order_docs = stage.ranked

        if collapse_duplicates:                                   # keep the best-ranked of identical (title, employer, cities) postings
            seen, keep = set(), []
            for d in order_docs:
                d = int(d)
                key = (str(ix.meta["title"][d]).lower(), str(ix.meta["company"][d] or "").lower(), tuple(ix.meta["cities"][d] or [ix.meta["city"][d]]))
                if key not in seen:
                    seen.add(key)
                    keep.append(d)
                if len(keep) >= k:
                    break
            top_docs = np.array(keep, dtype=np.int64)
        else:
            top_docs = np.asarray(order_docs[:k], dtype=np.int64)
        t3 = time.perf_counter()

        # ---- explanations for the displayed results ------------------------------------------------------------
        need_text = [d for d in top_docs if int(d) not in passage_info] if (explain and not browse) else []
        if need_text:
            passage_info.update(self._passages(np.array(need_text), qterms))
        stems = set(parsed.terms)
        q_tids = {t: ix.terms[t] for t, _ in qterms}
        for rank, d in enumerate(top_docs, 1):
            d = int(d)
            m = ix.meta
            doc_skills = set(m["skills"][d] or [])
            reasons: List[str] = []
            raw_scores: Dict[str, float] = {}
            if d in stage.lm:
                raw_scores["lm_loglik_per_term"] = float(stage.lm[d] / total_q)
            if d in stage.dfr:
                raw_scores["dfr"] = float(stage.dfr[d])
            if ix.pagerank is not None:
                raw_scores["pagerank"] = float(ix.pagerank[d])
            bp = passage_info.get(d)
            passage = None
            if bp and bp[0] is not None:
                p = bp[0]
                passage = {"text": p.text, "html": PS.highlight(p.text, stems), "section": p.section, "score": None if not math.isfinite(bp[1]) else round(float(bp[1]), 4)}
                if explain:
                    raw_scores["passage_llr"] = float(bp[1]) if math.isfinite(bp[1]) else float("nan")
            if explain and not browse:
                tokens = set(ix.doc_tokens(d).tolist())
                for tid, term in q_tids.items():
                    surf = parsed.surface.get(term, term)
                    reasons.append(f"✓ '{surf}' matched" if tid in tokens else f"✗ '{surf}' not in this posting")
                if passage:
                    reasons.append("✓ relevant passage found" + (f" (section: {passage['section']})" if passage["section"] else ""))
            if parsed.skills:
                _, hit, miss = skill_score(doc_skills, parsed.skills)
                reasons += [f"✓ skill: {s}" for s in hit] + [f"✗ skill missing: {s}" for s in miss]
            if parsed.locations:
                ls = location_score(m["cities"][d], m["city"][d], parsed.locations)
                reasons.append(f"✓ location matches {', '.join(parsed.locations)}" if ls == 1.0 else ("? location unknown" if not (m["cities"][d] or m["city"][d]) else f"✗ location is {m['city'][d]}"))
            if parsed.experience:
                es = experience_score(m["exp_min"][d], m["exp_max"][d], parsed.experience["min"], parsed.experience["max"])
                reasons.append("✓ experience band fits" if es == 1.0 else ("? experience not stated" if math.isnan(m["exp_min"][d]) else "✗ experience band differs"))
            resp.results.append(SearchResult(
                rank=rank, job_id=int(ix.job_ids[d]), title=m["title"][d], company=m["company"][d], city=m["city"][d], cities=list(m["cities"][d] or []), country=m["country"][d],
                source=m["source"][d], score=raw_final[d] if raw_final is not None else stage.score.get(d, 0.0), signals=sig_at.get(d, {}), raw=raw_scores, passage=passage,
                reasons=reasons, experience=_fmt_exp(m["exp_min"][d], m["exp_max"][d]), salary=_fmt_salary(m["sal_min"][d], m["sal_max"][d], m["sal_cur"][d], m["sal_period"][d]),
                skills=sorted(doc_skills)[:12], text_quality=m["text_quality"][d]))
        t4 = time.perf_counter()
        resp.timings_ms = {"parse+filter": round((t1 - t0) * 1000, 1), "retrieve": round((t2 - t1) * 1000, 1), "passages": round(t_pass * 1000, 1),
                           "rank": round((t3 - t2 - t_pass) * 1000, 1), "explain": round((t4 - t3) * 1000, 1), "total": round((t4 - t0) * 1000, 1), **stage.extra}
        return resp
