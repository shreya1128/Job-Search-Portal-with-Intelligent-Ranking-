"""MapReduce jobs.
  market job          job -> skills; counts overall, by role cluster, city, experience band, month; skill pairs; distributions
  inverted-index job  document -> (term, (doc, tf)); reduce -> postings list.   Used to show that the retrieval index itself can be
                      built with MapReduce, and verified against the real index."""
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from jobintel.mapreduce.framework import sum_reducer

VETTED = {"technical", "tool", "knowledge", "certification"}     # skill types used for role/city/experience/pair tables (soft & 'other' only in the overall list)
PAIR_TYPES = {"technical", "tool"}
EXP_BANDS = [(0, 1, "0-1 yrs"), (1, 3, "1-3 yrs"), (3, 5, "3-5 yrs"), (5, 8, "5-8 yrs"), (8, 10 ** 6, "8+ yrs")]


def exp_band(x: Optional[float]) -> Optional[str]:
    if x is None:
        return None
    for lo, hi, name in EXP_BANDS:
        if lo <= x < hi:
            return name
    return None


def sum_combiner(key, values):
    return sum(values)


# =========================================================================== market analysis
def market_mapper(rec: Dict):
    """rec = {job_id, source, cluster, cities[], exp_min, month, company, category, salary_min, salary_cur, salary_period, skills[(name, type)]}"""
    yield ("jobs",), 1
    band, month = exp_band(rec.get("exp_min")), rec.get("month")
    yield ("dist", "cluster", rec["cluster"]), 1
    yield ("dist", "source", rec["source"]), 1
    if band:
        yield ("dist", "exp_band", band), 1
    if month:
        yield ("dist", "month", month), 1
    if rec.get("company"):
        yield ("dist", "company", rec["company"]), 1
    if rec.get("category"):
        yield ("dist", "category", rec["category"]), 1
    for c in rec.get("cities") or []:
        yield ("dist", "city", c), 1
    if rec.get("salary_min") and rec.get("salary_cur") == "SGD" and rec.get("salary_period") == "month":
        yield ("dist", "salary_sgd_month", int(rec["salary_min"] // 1000) * 1000), 1
    vetted = []
    for name, typ in rec["skills"]:
        yield ("skill", typ, name), 1
        if typ in VETTED:
            vetted.append(name)
            yield ("cluster_skill", rec["cluster"], name), 1
            for c in rec.get("cities") or []:
                yield ("city_skill", c, name), 1
            if band:
                yield ("exp_skill", band, name), 1
            if month:
                yield ("month_skill", month, name), 1
    tech = sorted({n for n, t in rec["skills"] if t in PAIR_TYPES})
    for a, b in itertools.combinations(tech, 2):
        yield ("pair", a, b), 1


def assemble_market(out: Dict, top: int = 15) -> Dict:
    n = out.get(("jobs",), 0)
    grouped: Dict[str, Dict] = defaultdict(lambda: defaultdict(dict))
    flat: Dict[str, Dict] = defaultdict(dict)
    pairs, skill_total = {}, defaultdict(int)
    for key, v in out.items():
        kind = key[0]
        if kind == "skill":
            flat["skill_by_type"].setdefault(key[1], {})[key[2]] = v
            if key[1] in VETTED:
                skill_total[key[2]] += v
        elif kind in ("cluster_skill", "city_skill", "exp_skill", "month_skill"):
            grouped[kind].setdefault(key[1], {})[key[2]] = v
        elif kind == "pair":
            pairs[(key[1], key[2])] = v
        elif kind == "dist":
            flat["dist"].setdefault(key[1], {})[key[2]] = v
    topn = lambda d, k=top: sorted(d.items(), key=lambda x: (-x[1], x[0]))[:k]
    pair_rows = []
    for (a, b), c in sorted(pairs.items(), key=lambda x: (-x[1], x[0]))[:300]:
        if c >= 20 and skill_total[a] and skill_total[b]:
            pair_rows.append({"a": a, "b": b, "jobs": c, "lift": round(c * n / (skill_total[a] * skill_total[b]), 2)})
    city_volume = flat["dist"].get("city", {})
    return {
        "total_jobs": n,
        "top_skills_by_type": {t: topn(d, 40) for t, d in flat["skill_by_type"].items()},
        "skills_by_cluster": {c: topn(d) for c, d in grouped["cluster_skill"].items()},
        "skills_by_city": {c: topn(grouped["city_skill"][c], 10) for c, _ in topn(city_volume, 12) if c in grouped["city_skill"]},
        "skills_by_experience": {b: topn(d, 10) for b, d in grouped["exp_skill"].items()},
        "skills_by_month": {m: {"jobs": flat["dist"].get("month", {}).get(m, 0), "top": topn(d, 10)} for m, d in grouped["month_skill"].items()},
        "top_pairs_by_count": pair_rows[:40],
        "top_pairs_by_lift": sorted([r for r in pair_rows if r["jobs"] >= 50], key=lambda r: -r["lift"])[:30],
        "distributions": {k: dict(topn(v, 25)) if k in ("company", "category", "city", "month") else dict(sorted(v.items())) if k in ("salary_sgd_month",) else dict(topn(v, 30)) for k, v in flat["dist"].items()},
    }


# =========================================================================== inverted index via MapReduce
def index_mapper(doc: Tuple[int, List[str]]):
    """doc = (doc_id, token list)  ->  (term, (doc_id, tf))  for every distinct term"""
    doc_id, tokens = doc
    tf: Dict[str, int] = defaultdict(int)
    for t in tokens:
        tf[t] += 1
    for t, f in tf.items():
        yield t, (doc_id, f)


def index_reducer(term, postings):
    postings.sort()
    return postings
