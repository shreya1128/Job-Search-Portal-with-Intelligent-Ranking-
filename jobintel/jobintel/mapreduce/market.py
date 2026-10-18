"""Feeds the market-analysis MapReduce job from the database, and the SQL queries used to VALIDATE its output."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Tuple

from sqlalchemy import text

WHERE = "j.duplicate_of IS NULL AND j.is_fraudulent = 0 AND j.text_quality <> 'none' AND s.source_type = 'dataset'"


def load_market_records(engine) -> List[Dict]:
    skills: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    with engine.connect() as c:
        for jid, name, typ in c.execute(text(f"""SELECT js.job_id, k.canonical_name, k.skill_type FROM job_skills js JOIN skills k USING(skill_id)
                JOIN jobs j USING(job_id) JOIN crawler_sources s USING(source_id) WHERE {WHERE} AND k.skill_type <> 'education'""")):
            skills[jid].append((name, typ))
        rows = c.execute(text(f"""SELECT j.job_id, s.name, j.cluster, j.cities, j.city, j.experience_min, j.posting_date, co.name, j.category, j.salary_min,
                j.salary_currency, j.salary_period FROM jobs j JOIN crawler_sources s USING(source_id) LEFT JOIN companies co USING(company_id) WHERE {WHERE}""")).all()
    out = []
    for jid, src, cl, cities, city, emin, d, comp, cat, smin, scur, sper in rows:
        cities = json.loads(cities) if isinstance(cities, str) else cities
        out.append(dict(job_id=jid, source=src, cluster=cl or "other", cities=list(cities or ([city] if city else [])), exp_min=emin,
                        month=d.strftime("%Y-%m") if d else None, company=comp, category=cat, salary_min=smin, salary_cur=scur, salary_period=sper,
                        skills=skills.get(jid, [])))
    return out


def sql_skill_counts(engine) -> Dict[Tuple[str, str], int]:
    with engine.connect() as c:
        return {(n, t): k for n, t, k in c.execute(text(f"""SELECT k.canonical_name, k.skill_type, COUNT(DISTINCT js.job_id) FROM job_skills js
                JOIN skills k USING(skill_id) JOIN jobs j USING(job_id) JOIN crawler_sources s USING(source_id)
                WHERE {WHERE} AND k.skill_type <> 'education' GROUP BY k.canonical_name, k.skill_type"""))}


def sql_scalar(engine, sql: str, **kw) -> int:
    with engine.connect() as c:
        return c.execute(text(sql), kw).scalar_one()
