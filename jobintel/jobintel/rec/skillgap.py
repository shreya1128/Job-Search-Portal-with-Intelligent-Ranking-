"""Skill-gap analysis: which skills a target job / role asks for that the profile lacks, ordered by a stated, UNVALIDATED importance heuristic.

importance(skill) = a * market_frequency + b * listed_in_the_job + c * essential_in_the_knowledge_graph        (defaults a=0.5, b=0.3, c=0.2; all terms in [0, 1])
  market_frequency  share of SIMILAR jobs (the 50 nearest jobs in skill space for a job target; the jobs whose title contains the role words for a role target) that list it
  listed_in_the_job 1 when the target job itself lists the skill (job targets only)
  essential         1 when the best-matching occupation of the knowledge graph requires it
The weights are a transparent ordering heuristic, not learned and not validated against hiring outcomes.  `skill_coverage` = importance-weighted share of the target's skills the profile
has - a descriptive number, NOT a probability of getting the job.  A missing skill for which the profile holds a related skill (knowledge graph) is reported with that skill, as a bridge."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULT_IMPORTANCE = (0.5, 0.3, 0.2)


@dataclass
class GapRow:
    skill: str
    status: str                                   # 'matching' | 'missing'
    importance: float
    market_frequency: float
    listed_in_job: bool
    kb_essential: bool
    related_held: List[str] = field(default_factory=list)


@dataclass
class GapReport:
    target: str
    rows: List[GapRow]
    skill_coverage: Optional[float]
    n_similar_jobs: int
    occupation: Optional[str]
    warnings: List[str] = field(default_factory=list)

    @property
    def missing(self) -> List[GapRow]:
        return [r for r in self.rows if r.status == "missing"]

    @property
    def matching(self) -> List[GapRow]:
        return [r for r in self.rows if r.status == "matching"]

    def to_dict(self) -> Dict:
        return {**asdict(self), "rows": [asdict(r) for r in self.rows]}


def analyze_gap(catalog, kg, profile_weights: Dict[str, float], *, job_di: Optional[int] = None, role_text: Optional[str] = None, n_similar: int = 50,
                min_market_freq: float = 0.25, importance=DEFAULT_IMPORTANCE, max_rows: int = 40) -> GapReport:
    if (job_di is None) == (role_text is None):
        raise ValueError("give exactly one of job_di / role_text")
    a, b, c = importance
    warnings: List[str] = []
    if job_di is not None:
        title = catalog.ix.meta["title"][job_di]
        listed = set(catalog.job_skills(job_di))
        sim = catalog.similar_jobs(job_di, n_similar)
        target = f"job {int(catalog.ix.job_ids[job_di])}: {title}"
        occ_text = title
    else:
        listed = set()
        sim = catalog.jobs_by_title(role_text, limit=max(n_similar, 200))
        target = f"role: {role_text}"
        occ_text = role_text
        if not len(sim):
            warnings.append("no job title contains the words of this role: market frequencies unavailable")
    if len(sim) < 5:
        warnings.append(f"only {len(sim)} similar jobs: market frequencies are unreliable")
    freq = catalog.market_frequency(sim) if len(sim) else {}
    occs = kg.match_title(occ_text, top=1) if kg is not None else []
    occ = occs[0]["occupation"] if occs else None
    essential = set(kg.essential_skills(occ)) if occ else set()
    cand = set(listed) | {s for s, f in freq.items() if f >= min_market_freq} | (essential & set(catalog.sidx))
    rows: List[GapRow] = []
    for s in cand:
        imp = a * freq.get(s, 0.0) + b * (1.0 if s in listed else 0.0) + c * (1.0 if s in essential else 0.0)
        have = s in profile_weights
        bridge = [] if have or kg is None else sorted(r for r in kg.related.get(s, {}) if r in profile_weights)[:3]
        rows.append(GapRow(s, "matching" if have else "missing", round(imp, 4), round(freq.get(s, 0.0), 4), s in listed, s in essential, bridge))
    rows.sort(key=lambda r: (-r.importance, r.skill))
    rows = rows[:max_rows]
    tot = sum(r.importance for r in rows)
    cov = (sum(r.importance for r in rows if r.status == "matching") / tot) if tot > 0 else None
    if not rows:
        warnings.append("no target skills could be determined")
    return GapReport(target, rows, None if cov is None else round(cov, 4), int(len(sim)), occ, warnings)
