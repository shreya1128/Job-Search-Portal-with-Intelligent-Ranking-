"""KnowledgeGraph: occupations <-> skills (ESCO / curated, plus O*NET when ingested) and skill <-> skill relations, as used by the knowledge-based recommender.

Edges (the Phase 1 tables): occupation -requires_essential-> skill (weight 1.0), occupation -requires_optional-> skill (0.5), skill -related_to-> skill (ESCO, weight ~0.4),
skill -broader_than-> skill (curated).  The graph is whatever was ingested: an absent O*NET / ESCO file simply means fewer occupations, never invented edges.

knowledge_scores() = for the occupations the profile points at (ranked by coverage of their essential/optional skills, plus any whose NAME matches the preferred role),
how much of each occupation's skill demand a job covers, averaged with the occupation weights.  It needs no interaction history, so it is the cold-start recommender."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

_TOK = re.compile(r"[a-z0-9+#.]+")
_STOP = {"and", "the", "of", "for", "a", "an", "in", "to", "senior", "junior", "lead", "sr", "jr", "i", "ii", "iii"}


def _tokens(text: str) -> set:
    return {t for t in _TOK.findall((text or "").lower()) if t not in _STOP and len(t) > 1}


class KnowledgeGraph:
    def __init__(self, occ_skills: Dict[str, Dict[str, Tuple[str, float]]], related: Dict[str, Dict[str, float]], occ_aliases: Optional[Dict[str, List[str]]] = None):
        self.occ_skills = occ_skills                      # occupation -> skill -> (kind 'essential'|'optional', weight)
        self.related = related                            # skill -> related skill -> weight (symmetric)
        self.occ_aliases = occ_aliases or {}
        self._occ_tokens = {o: [_tokens(o)] + [_tokens(a) for a in self.occ_aliases.get(o, [])] for o in occ_skills}

    # ------------------------------------------------------------------------------------------------ construction
    @classmethod
    def from_dicts(cls, occupations: Dict[str, Dict[str, Iterable[str]]], related: Optional[Dict[str, Dict[str, float]]] = None) -> "KnowledgeGraph":
        occ = {o: {**{s: ("essential", 1.0) for s in d.get("essential", [])}, **{s: ("optional", 0.5) for s in d.get("optional", []) if s not in d.get("essential", [])}} for o, d in occupations.items()}
        rel: Dict[str, Dict[str, float]] = defaultdict(dict)
        for a, d in (related or {}).items():
            for b, w in d.items():
                rel[a][b] = max(w, rel[a].get(b, 0)); rel[b][a] = max(w, rel[b].get(a, 0))
        return cls(occ, dict(rel))

    @classmethod
    def from_db(cls, engine) -> "KnowledgeGraph":
        import pandas as pd
        sk = pd.read_sql("SELECT skill_id, canonical_name FROM skills", engine)
        oc = pd.read_sql("SELECT occupation_id, name, aliases FROM occupations", engine)
        rel = pd.read_sql("SELECT subject_type, subject_id, relation, object_type, object_id, weight FROM knowledge_relationships", engine)
        sname = dict(zip(sk.skill_id, sk.canonical_name))
        oname = dict(zip(oc.occupation_id, oc["name"]))
        occ_skills: Dict[str, Dict[str, Tuple[str, float]]] = defaultdict(dict)
        related: Dict[str, Dict[str, float]] = defaultdict(dict)
        for r in rel.itertuples(index=False):
            if r.subject_type == "occupation" and r.object_type == "skill" and r.relation in ("requires_essential", "requires_optional"):
                o, s = oname.get(r.subject_id), sname.get(r.object_id)
                if o and s:
                    kind = "essential" if r.relation == "requires_essential" else "optional"
                    if s not in occ_skills[o] or kind == "essential":
                        occ_skills[o][s] = (kind, 1.0 if kind == "essential" else 0.5)
            elif r.subject_type == "skill" and r.object_type == "skill" and r.relation in ("related_to", "broader_than"):
                a, b = sname.get(r.subject_id), sname.get(r.object_id)
                if a and b and a != b:
                    w = float(r.weight)
                    related[a][b] = max(w, related[a].get(b, 0)); related[b][a] = max(w, related[b].get(a, 0))
        aliases = {oname[i]: list(a) for i, a in zip(oc.occupation_id, oc.aliases) if isinstance(a, list) and a and i in oname}
        return cls(dict(occ_skills), dict(related), aliases)

    # ------------------------------------------------------------------------------------------------ queries
    def expand(self, weights: Dict[str, float], decay: float = 0.5, per_skill: int = 5) -> Tuple[Dict[str, float], Dict[str, str]]:
        """add the strongest related skills (weight * decay * edge weight) that the profile does not already have.  Returns (weights, {added skill: because of})."""
        out, added = dict(weights), {}
        for s, w in weights.items():
            for r, ew in sorted(self.related.get(s, {}).items(), key=lambda x: -x[1])[:per_skill]:
                if r not in weights and w * decay * ew > out.get(r, 0.0):
                    out[r], added[r] = w * decay * ew, s
        return out, added

    def rank_occupations(self, weights: Dict[str, float], top: int = 5, min_matched: int = 2) -> List[Dict]:
        """occupations ranked by the share of their (essential 1.0 / optional 0.5) skill demand that the profile covers."""
        res = []
        for o, sk in self.occ_skills.items():
            tot = sum(0.5 if k == "optional" else 1.0 for k, _ in sk.values())
            got = [(s, min(weights[s], 1.0) * (0.5 if sk[s][0] == "optional" else 1.0)) for s in sk if s in weights]
            if len(got) < min_matched or tot <= 0:
                continue
            res.append({"occupation": o, "score": sum(v for _, v in got) / tot, "matched": sorted(s for s, _ in got),
                        "missing_essential": sorted(s for s, (k, _) in sk.items() if k == "essential" and s not in weights)})
        return sorted(res, key=lambda r: -r["score"])[:top]

    def match_title(self, text: str, top: int = 3, min_overlap: float = 0.5) -> List[Dict]:
        """occupations whose name / alias shares at least `min_overlap` of its words with the text (Jaccard-like, cheap and explainable)."""
        q = _tokens(text)
        if not q:
            return []
        res = []
        for o, variants in self._occ_tokens.items():
            best = max((len(q & v) / len(v) if v else 0.0) for v in variants)
            if best >= min_overlap and any(q & v for v in variants):
                res.append({"occupation": o, "score": best})
        return sorted(res, key=lambda r: (-r["score"], len(r["occupation"])))[:top]

    def essential_skills(self, occupation: str) -> List[str]:
        return sorted(s for s, (k, _) in self.occ_skills.get(occupation, {}).items() if k == "essential")

    # ------------------------------------------------------------------------------------------------ the recommender score
    def knowledge_scores(self, catalog, weights: Dict[str, float], preferred_role: Optional[str] = None, top_occ: int = 5) -> Tuple[Optional[np.ndarray], Dict]:
        exp_w, added = self.expand(weights)
        ranked = self.rank_occupations(exp_w, top=top_occ)
        occ_w: Dict[str, float] = {}
        if ranked:
            m = ranked[0]["score"] or 1.0
            occ_w.update({r["occupation"]: r["score"] / m for r in ranked})
        titled = self.match_title(preferred_role, top=3) if preferred_role else []
        for r in titled:
            occ_w[r["occupation"]] = 1.0                    # an explicit role is intent: it gets the full weight
        info = {"occupations": [{"occupation": o, "weight": round(w, 3), "from": ("preferred role" if any(t["occupation"] == o for t in titled) else "profile skills")} for o, w in sorted(occ_w.items(), key=lambda x: -x[1])],
                "related_skills_added": added}
        if not occ_w:
            info["reason"] = "no occupation matches the profile skills or the preferred role"
            return None, info
        names = list(occ_w)
        rows, cols, vals = [], [], []
        for j, o in enumerate(names):
            for s, (k, w) in self.occ_skills[o].items():
                i = catalog.sidx.get(s)
                if i is not None:
                    rows.append(i); cols.append(j); vals.append(w)
        if not vals:
            info["reason"] = "the matched occupations have no skill that occurs in the job catalogue"
            return None, info
        O = sp.csr_matrix((vals, (rows, cols)), shape=(len(catalog.skills), len(names)))
        tot = np.asarray(O.sum(axis=0)).ravel()
        cov = np.asarray((catalog.X @ O).todense()) / np.maximum(tot, 1e-12)             # (jobs x occupations): share of the occupation's skill demand the job lists
        ow = np.array([occ_w[o] for o in names])
        score = (cov * ow).sum(axis=1) / ow.sum()
        score = np.where(catalog.has_skills, score, 0.0)
        return score, info
