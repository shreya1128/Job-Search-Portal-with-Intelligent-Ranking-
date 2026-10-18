#!/usr/bin/env python3
"""Phase-1 test gate: integrity + quality report of whatever is in the database. Writes docs/phase1_report.md."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import text
from jobintel.config import ROOT
from jobintel.db.models import ALL_TABLES
from jobintel.db.session import get_engine

e = get_engine()
out, fails = [], []


def q(sql, **kw):
    with e.connect() as c:
        return c.execute(text(sql), kw).all()


def table(title, rows, header):
    out.append(f"\n### {title}\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for r in rows:
        out.append("| " + " | ".join("" if v is None else (f"{v:,.1f}" if isinstance(v, float) else str(v)) for v in r) + " |")


def check(name, sql, expect_zero=True):
    n = q(sql)[0][0]
    ok = (n == 0) if expect_zero else (n > 0)
    out.append(f"- {'PASS' if ok else 'FAIL'} - {name} ({n})")
    if not ok:
        fails.append(name)


out.append("# Phase 1 verification report\n")
table("Row counts", [(t, q(f"SELECT COUNT(*) FROM `{t}`")[0][0]) for t in ALL_TABLES], ["table", "rows"])
table("Jobs per source", q("""SELECT s.name, s.source_type, COUNT(*) jobs, SUM(j.duplicate_of IS NULL) unique_jobs, ROUND(100*AVG(j.duplicate_of IS NOT NULL),1) dup_pct,
    ROUND(AVG(j.n_tokens)) avg_tokens, ROUND(100*AVG(j.experience_min IS NOT NULL),0) exp_pct, ROUND(100*AVG(j.salary_min IS NOT NULL),0) salary_pct,
    ROUND(100*AVG(j.education IS NOT NULL),0) edu_pct, ROUND(100*AVG(j.category IS NOT NULL),0) label_pct, MIN(j.posting_date) first_date, MAX(j.posting_date) last_date
    FROM jobs j JOIN crawler_sources s USING(source_id) GROUP BY s.name, s.source_type ORDER BY jobs DESC"""),
      ["source", "type", "jobs", "unique", "dup %", "avg tokens", "exp %", "salary %", "edu %", "label %", "first date", "last date"])
table("Text quality", q("SELECT s.name, j.text_quality, COUNT(*) FROM jobs j JOIN crawler_sources s USING(source_id) GROUP BY s.name, j.text_quality ORDER BY s.name"), ["source", "text_quality", "jobs"])
table("Title-keyword clusters (heuristic, NOT ground truth)", q("SELECT cluster, COUNT(*) FROM jobs WHERE duplicate_of IS NULL GROUP BY cluster ORDER BY 2 DESC"), ["cluster", "unique jobs"])
table("Top technical skills (unique jobs; dataset tags + extracted)", q("""SELECT s.canonical_name, COUNT(*) n FROM job_skills js JOIN skills s USING(skill_id) JOIN jobs j USING(job_id)
    WHERE j.duplicate_of IS NULL AND s.skill_type IN ('technical','tool') AND s.ambiguous=0 GROUP BY s.canonical_name ORDER BY n DESC LIMIT 15"""), ["skill", "jobs"])
table("Skill origin", q("SELECT origin, COUNT(*) FROM job_skills GROUP BY origin"), ["origin", "links"])
table("Knowledge graph", q("SELECT source, relation, COUNT(*) FROM knowledge_relationships GROUP BY source, relation ORDER BY source, 3 DESC"), ["source", "relation", "edges"])

out.append("\n### Integrity checks\n")
check("jobs without a source", "SELECT COUNT(*) FROM jobs j LEFT JOIN crawler_sources s USING(source_id) WHERE s.source_id IS NULL")
check("indexable jobs (non-duplicate, has text) missing processed_text", "SELECT COUNT(*) FROM jobs WHERE duplicate_of IS NULL AND text_quality<>'none' AND (processed_text IS NULL OR processed_text='')")
check("original description lost (full-text job with NULL description_original)", "SELECT COUNT(*) FROM jobs WHERE text_quality IN ('full','preprocessed') AND description_original IS NULL")
check("experience_min > experience_max", "SELECT COUNT(*) FROM jobs WHERE experience_min > experience_max")
check("duplicate_of pointing to a missing job", "SELECT COUNT(*) FROM jobs a LEFT JOIN jobs b ON a.duplicate_of=b.job_id WHERE a.duplicate_of IS NOT NULL AND b.job_id IS NULL")
check("job is its own duplicate", "SELECT COUNT(*) FROM jobs WHERE duplicate_of=job_id")
check("knowledge edge: subject does not exist", """SELECT COUNT(*) FROM knowledge_relationships r WHERE (r.subject_type='skill' AND NOT EXISTS (SELECT 1 FROM skills s WHERE s.skill_id=r.subject_id))
    OR (r.subject_type='occupation' AND NOT EXISTS (SELECT 1 FROM occupations o WHERE o.occupation_id=r.subject_id))""")
check("knowledge edge: object does not exist", """SELECT COUNT(*) FROM knowledge_relationships r WHERE (r.object_type='skill' AND NOT EXISTS (SELECT 1 FROM skills s WHERE s.skill_id=r.object_id))
    OR (r.object_type='occupation' AND NOT EXISTS (SELECT 1 FROM occupations o WHERE o.occupation_id=r.object_id))""")
check("skills linked to jobs but ambiguous single letters", "SELECT COUNT(*) FROM job_skills js JOIN skills s USING(skill_id) JOIN jobs j USING(job_id) WHERE j.text_quality='full' AND js.origin='extracted' AND CHAR_LENGTH(s.canonical_name)=1")
check("database has job rows", "SELECT COUNT(*) FROM jobs", expect_zero=False)
check("knowledge base populated", "SELECT COUNT(*) FROM knowledge_relationships", expect_zero=False)
out.append(f"\n**Result: {'ALL CHECKS PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}**")
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "phase1_report.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
sys.exit(1 if fails else 0)
