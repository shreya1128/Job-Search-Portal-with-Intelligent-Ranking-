"""Skill / occupation knowledge base loaders: curated graph, ESCO (CSV) and O*NET (Excel) -> skills, occupations, knowledge_relationships."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy import insert, select
from sqlalchemy.engine import Engine

from jobintel.db.models import KnowledgeRelationship as KR, Occupation, Skill
from jobintel.extraction.lexicon import SkillLexicon, load_lexicon, normalize_tag
from jobintel.ingest import curated_kb


def _clean(v) -> Optional[str]:
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else (str(v).strip() or None)


class KBWriter:
    def __init__(self, engine: Engine, lex: SkillLexicon):
        self.engine, self.lex = engine, lex
        self._dialect = engine.dialect.name
        with engine.connect() as c:
            self.skill_ids: Dict[str, int] = {n.lower(): i for i, n in c.execute(select(Skill.skill_id, Skill.canonical_name))}
            self.occ_ids: Dict[tuple, int] = {(s, e): i for i, s, e in c.execute(select(Occupation.occupation_id, Occupation.source, Occupation.external_id))}

    def _ins(self, table):
        return insert(table).prefix_with("OR IGNORE", dialect="sqlite").prefix_with("IGNORE", dialect="mysql")

    def canon(self, label: str) -> Optional[str]:
        label = re.sub(r"\s+(software|platform|tool|tools|application)$", "", str(label).strip(), flags=re.I)
        c = self.lex.normalize(label)
        return c[:200] if c else None

    def ensure_skills(self, items: Dict[str, dict]) -> None:
        """items: canonical -> dict(skill_type, source, aliases, esco_uri, ambiguous). Inserts the unknown ones."""
        new = {k: v for k, v in items.items() if k.lower() not in self.skill_ids}
        if not new:
            return
        rows = [dict(name=k[:200], canonical_name=k[:200], skill_type=v.get("skill_type", "other"), source=v.get("source", "kb"),
                     aliases=v.get("aliases") or None, esco_uri=v.get("esco_uri"), ambiguous=bool(v.get("ambiguous", False))) for k, v in new.items()]
        with self.engine.begin() as c:
            for i in range(0, len(rows), 2000):
                c.execute(self._ins(Skill), rows[i:i + 2000])
            keys = list(new)
            for i in range(0, len(keys), 500):
                for sid, cn in c.execute(select(Skill.skill_id, Skill.canonical_name).where(Skill.canonical_name.in_(keys[i:i + 500]))):
                    self.skill_ids[cn.lower()] = sid

    def sid(self, canonical: str) -> Optional[int]:
        return self.skill_ids.get(canonical.lower())

    def ensure_occupations(self, rows: List[dict]) -> None:
        new = [r for r in rows if (r["source"], r["external_id"]) not in self.occ_ids]
        if not new:
            return
        with self.engine.begin() as c:
            for i in range(0, len(new), 2000):
                c.execute(self._ins(Occupation), new[i:i + 2000])
            for src in {r["source"] for r in new}:
                for oid, ext in c.execute(select(Occupation.occupation_id, Occupation.external_id).where(Occupation.source == src)):
                    self.occ_ids[(src, ext)] = oid

    def edges(self, rows: List[dict]) -> int:
        if not rows:
            return 0
        with self.engine.begin() as c:
            for i in range(0, len(rows), 5000):
                c.execute(self._ins(KR), rows[i:i + 5000])
        return len(rows)


def _edge(st, sid, rel, ot, oid, w, src):
    return dict(subject_type=st, subject_id=sid, relation=rel, object_type=ot, object_id=oid, weight=w, source=src)


# =========================================================================== curated
def load_curated(engine: Engine, lex: Optional[SkillLexicon] = None) -> Dict:
    lex = lex or load_lexicon()
    w = KBWriter(engine, lex)
    names = set(curated_kb.HIERARCHY)
    for subs in curated_kb.HIERARCHY.values():
        names.update(subs)
    for ess, opt in curated_kb.ROLES.values():
        names.update(ess + opt)
    canon = {n: (lex.normalize(n) or normalize_tag(n)) for n in names}
    w.ensure_skills({c: dict(skill_type=(lex.get(c).type if lex.get(c) else "technical"), source="curated") for c in canon.values()})
    occ_rows, edges = [], []
    for role in curated_kb.ROLES:
        occ_rows.append(dict(name=role, source="curated", external_id=re.sub(r"\W+", "-", role.lower())))
    w.ensure_occupations(occ_rows)
    oid = lambda role: w.occ_ids[("curated", re.sub(r"\W+", "-", role.lower()))]
    for parent, subs in curated_kb.HIERARCHY.items():
        p = w.sid(canon[parent])
        for s in subs:
            if w.sid(canon[s]) and p != w.sid(canon[s]):
                edges.append(_edge("skill", p, "broader_than", "skill", w.sid(canon[s]), 1.0, "curated"))
    for role, (ess, opt) in curated_kb.ROLES.items():
        for s in ess:
            edges.append(_edge("occupation", oid(role), "requires_essential", "skill", w.sid(canon[s]), 1.0, "curated"))
        for s in opt:
            edges.append(_edge("occupation", oid(role), "requires_optional", "skill", w.sid(canon[s]), 0.5, "curated"))
    for a, b in curated_kb.SIMILAR:
        edges.append(_edge("occupation", oid(a), "related_to", "occupation", oid(b), 0.8, "curated"))
        edges.append(_edge("occupation", oid(b), "related_to", "occupation", oid(a), 0.8, "curated"))
    return dict(source="curated", occupations=len(occ_rows), edges=w.edges(edges))


# =========================================================================== ESCO
def load_esco(engine: Engine, esco_dir: Path, lex: Optional[SkillLexicon] = None) -> Dict:
    lex = lex or load_lexicon()
    d = Path(esco_dir)
    w = KBWriter(engine, lex)
    t0 = time.time()
    digital = set(pd.read_csv(d / "digitalSkillsCollection_en.csv", usecols=["conceptUri"])["conceptUri"]) if (d / "digitalSkillsCollection_en.csv").exists() else set()
    sk = pd.read_csv(d / "skills_en.csv", usecols=["conceptUri", "skillType", "preferredLabel", "altLabels"])
    uri2canon: Dict[str, str] = {}
    items: Dict[str, dict] = {}
    for r in sk.itertuples(index=False):
        label = _clean(r.preferredLabel)
        if not label:
            continue
        c = w.canon(label)
        if not c:
            continue
        uri2canon[r.conceptUri] = c
        if c.lower() not in items and c.lower() not in w.skill_ids:
            alts = [a.strip() for a in str(r.altLabels).split("\n") if a.strip() and a != "nan"][:8]
            typ = "technical" if r.conceptUri in digital else ("knowledge" if r.skillType == "knowledge" else "other")
            items[c] = dict(skill_type=typ, source="esco", aliases=alts, esco_uri=r.conceptUri[:300])
    w.ensure_skills(items)
    occ = pd.read_csv(d / "occupations_en.csv", usecols=["conceptUri", "preferredLabel", "altLabels", "description", "iscoGroup"])
    rows = []
    for r in occ.itertuples(index=False):
        alts = [a.strip() for a in str(r.altLabels).split("\n") if a.strip() and a != "nan"][:12]
        rows.append(dict(name=str(r.preferredLabel)[:300], source="esco", external_id=r.conceptUri[:300], description=(_clean(r.description) or "")[:2000] or None, aliases=alts or None))
    w.ensure_occupations(rows)
    edges: List[dict] = []
    osr = pd.read_csv(d / "occupationSkillRelations_en.csv", usecols=["occupationUri", "relationType", "skillUri"])
    for r in osr.itertuples(index=False):
        o, s = w.occ_ids.get(("esco", r.occupationUri[:300])), uri2canon.get(r.skillUri)
        sid = w.sid(s) if s else None
        if o and sid:
            ess = r.relationType == "essential"
            edges.append(_edge("occupation", o, "requires_essential" if ess else "requires_optional", "skill", sid, 1.0 if ess else 0.5, "esco"))
    n_os = len(edges)
    if (d / "skillSkillRelations_en.csv").exists():
        for r in pd.read_csv(d / "skillSkillRelations_en.csv", usecols=["originalSkillUri", "relationType", "relatedSkillUri"]).itertuples(index=False):
            a, b = uri2canon.get(r.originalSkillUri), uri2canon.get(r.relatedSkillUri)
            if a and b and w.sid(a) and w.sid(b) and w.sid(a) != w.sid(b):
                edges.append(_edge("skill", w.sid(a), "related_to", "skill", w.sid(b), 0.7 if r.relationType == "essential" else 0.4, "esco"))
    if digital:
        dg = pd.read_csv(d / "digitalSkillsCollection_en.csv", usecols=["conceptUri", "broaderConceptUri"]).dropna()
        for r in dg.itertuples(index=False):
            a, b = uri2canon.get(r.broaderConceptUri), uri2canon.get(r.conceptUri)
            if a and b and w.sid(a) and w.sid(b) and w.sid(a) != w.sid(b):
                edges.append(_edge("skill", w.sid(a), "broader_than", "skill", w.sid(b), 1.0, "esco"))
    return dict(source="esco", skills_seen=len(uri2canon), new_skills=len(items), occupations=len(rows), occ_skill_edges=n_os,
                total_edges=w.edges(edges), seconds=round(time.time() - t0, 1))


# =========================================================================== O*NET
def load_onet(engine: Engine, onet_dir: Path, lex: Optional[SkillLexicon] = None) -> Dict:
    lex = lex or load_lexicon()
    d = Path(onet_dir)
    w = KBWriter(engine, lex)
    t0 = time.time()
    xl = lambda name, **kw: pd.read_excel(d / f"{name}.xlsx", **kw)
    occ = xl("Occupation Data")
    zones = dict(zip(xl("Job Zones")["O*NET-SOC Code"], xl("Job Zones")["Job Zone"]))
    titles = xl("Sample of Reported Titles").groupby("O*NET-SOC Code")["Reported Job Title"].apply(lambda s: list(dict.fromkeys(s))[:15]).to_dict()
    rows = [dict(name=str(r["Title"])[:300], source="onet", external_id=str(r["O*NET-SOC Code"]), description=(_clean(r["Description"]) or "")[:2000] or None,
                 job_zone=int(zones[r["O*NET-SOC Code"]]) if r["O*NET-SOC Code"] in zones and pd.notna(zones[r["O*NET-SOC Code"]]) else None,
                 aliases=titles.get(r["O*NET-SOC Code"])) for _, r in occ.iterrows()]
    w.ensure_occupations(rows)
    sw = xl("Software Skills")
    items: Dict[str, dict] = {}
    pairs = []
    for r in sw.itertuples(index=False):
        hot, dem = r[5] == "Y", r[6] == "Y"
        c = w.canon(r[2])
        if not c or len(c) > 60:
            continue
        if not (hot or dem or c.lower() in w.skill_ids):
            continue
        if c.lower() not in w.skill_ids:
            items.setdefault(c, dict(skill_type="tool", source="onet"))
        pairs.append((str(r[0]), c, 1.0 if hot else (0.7 if dem else 0.4)))
    w.ensure_skills(items)
    edges, seen = [], set()
    for code, c, wt in pairs:
        o, s = w.occ_ids.get(("onet", code)), w.sid(c)
        if o and s and (o, s) not in seen:
            seen.add((o, s))
            edges.append(_edge("occupation", o, "uses_technology", "skill", s, wt, "onet"))
    n_tech = len(edges)
    tier = {"Primary-Short": 1.0, "Primary-Long": 0.8, "Supplemental": 0.5}
    for r in xl("Related Occupations").itertuples(index=False):
        a, b = w.occ_ids.get(("onet", str(r[0]))), w.occ_ids.get(("onet", str(r[2])))
        if a and b:
            edges.append(_edge("occupation", a, "related_to", "occupation", b, tier.get(r[4], 0.5), "onet"))
    return dict(source="onet", occupations=len(rows), new_skills=len(items), technology_edges=n_tech, total_edges=w.edges(edges), seconds=round(time.time() - t0, 1))
