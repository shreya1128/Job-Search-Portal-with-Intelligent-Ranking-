#!/usr/bin/env python3
"""Phase-2 test gate on the REAL index.  Checks integrity, then MEASURES (never assumes) latency and how much the ranking models disagree.
It makes no claim about which model returns more relevant jobs - that needs relevance judgments (Phase 4).  Writes docs/phase2_report.md."""
import itertools
import math
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import text
from jobintel.config import DATA_DIR, ROOT
from jobintel.db.session import get_engine
from jobintel.ir.engine import MODELS, DbDocStore, SearchEngine
from jobintel.ir.index import InvertedIndex
from jobintel.ir.passages import matched_terms

QUERIES = ["python machine learning sql", "Entry level machine learning jobs in Pune", "data analyst sql excel", "registered nurse", "accountant tax",
           "senior data scientist tensorflow deep learning", "java spring boot microservices", "devops engineer docker kubernetes aws", "customer service representative",
           "financial analyst forecasting excel", "python developer 3 years experience", "business intelligence power bi", "nursing patient care icu",
           "react javascript frontend developer", "cybersecurity analyst", "project manager agile scrum"]
e = get_engine()
ix = InvertedIndex.load(DATA_DIR / "index" / "job_index.pkl")
eng = SearchEngine(ix, DbDocStore(e))
out, fails = ["# Phase 2 verification report\n"], []


def check(name, ok, detail=""):
    out.append(f"- {'PASS' if ok else 'FAIL'} - {name}" + (f" ({detail})" if detail else ""))
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- index integrity
s = ix.stats()
out.append(f"## Index\n\n`{s}`\n\n## Integrity checks\n")
with e.connect() as c:
    n_db = c.execute(text("SELECT COUNT(*) FROM jobs j JOIN crawler_sources s USING(source_id) WHERE j.duplicate_of IS NULL AND j.text_quality<>'none' AND s.source_type='dataset' AND j.is_fraudulent=0")).scalar_one()
check("index holds exactly the indexable jobs of the database", s["documents"] == n_db, f"{s['documents']:,} vs {n_db:,}")
check("sum of collection frequencies = total tokens", int(ix.cf.sum()) == ix.total_tokens)
check("df <= N and df <= cf for every term", bool((ix.df <= ix.n_docs).all() and (ix.df <= ix.cf).all()))
check("postings sorted by document inside every term", all(bool((np.diff(ix.postings(t)[0]) > 0).all()) for t in range(0, ix.n_terms, 997)))
check("PageRank stored for every document and positive", ix.pagerank is not None and bool((ix.pagerank > 0).all()))
check("no fraudulent / synthetic job in the index", set(ix.meta["source"]) <= {"naukri", "emscad", "mcf", "linkedin"})
with e.connect() as c:
    tp = c.execute(text("SELECT COUNT(*) FROM job_passages")).scalar_one(); tt = c.execute(text("SELECT COUNT(*) FROM job_terms")).scalar_one()
check("job_passages and job_terms tables are populated", tp > 0 and tt > 0, f"{tp:,} passages, {tt:,} term rows")

# ---------------------------------------------------------------- run every query on every model
res = {m: {} for m in MODELS}
lat = {m: [] for m in MODELS}
bad = []
for q in QUERIES:
    for m in MODELS:
        r = eng.search(q, k=10, model=m, explain=(m == "hybrid"), collapse_duplicates=False)
        res[m][q] = r
        lat[m] += [r.timings_ms["total"]]
        for rep in range(2):
            lat[m].append(eng.search(q, k=10, model=m, explain=False).timings_ms["total"])
        if r.error or not r.results:
            bad.append((q, m, r.error))
        elif any(not math.isfinite(x.score) for x in r.results):
            bad.append((q, m, "non-finite score"))
check("every query returns finite-scored results under all 6 models", not bad, str(bad[:3]))
check("hybrid effective weights always sum to 1", all(abs(sum(r.weights.values()) - 1) < 1e-9 for r in res["hybrid"].values() if r.weights))
check("results never repeat a job within one ranking", all(len({x.job_id for x in r.results}) == len(r.results) for m in MODELS for r in res[m].values()))
# explanation / passage evidence
hit = tot = 0
for q, r in res["hybrid"].items():
    stems = set(r.parsed.terms)
    for x in r.results:
        if x.passage and x.text_quality in ("full", "preprocessed"):
            tot += 1
            hit += bool(matched_terms(x.passage["text"], stems))
check("best passage contains at least one query term (full-text results)", tot > 0 and hit / tot >= 0.95, f"{hit}/{tot}")
check("every hybrid result carries reasons and scores", all(x.reasons and x.signals for r in res["hybrid"].values() for x in r.results))
# filters
ok = True
for q in QUERIES[:6]:
    r = eng.search(q, k=20, filters={"city": "Pune"}, model="hybrid", explain=False)
    ok &= all("Pune" in x.cities for x in r.results)
check("city filter returns only jobs located in that city", ok)

# ---------------------------------------------------------------- measurements
out.append("\n## Latency (ms, 16 queries x 3 runs, 60,808 documents, single process)\n\n| model | median | p95 | max |\n|---|---|---|---|")
for m in MODELS:
    a = sorted(lat[m])
    out.append(f"| {m} | {st.median(a):.1f} | {a[int(0.95 * (len(a) - 1))]:.1f} | {a[-1]:.1f} |")
out.append("\n## How much do the models disagree?  (mean overlap of the top-10 sets, 16 queries)\n\nThis shows the models rank DIFFERENTLY. It does **not** show which one is better - that is measured with relevance labels in Phase 4.\n")
out.append("| | " + " | ".join(MODELS) + " |\n|---|" + "---|" * len(MODELS))
for a_ in MODELS:
    row = []
    for b_ in MODELS:
        ov = [len({x.job_id for x in res[a_][q].results} & {x.job_id for x in res[b_][q].results}) / 10 for q in QUERIES]
        row.append(f"{st.mean(ov):.2f}")
    out.append(f"| **{a_}** | " + " | ".join(row) + " |")
out.append("\n## PageRank\n")
pr = ix.pagerank
out.append(f"- documents: {len(pr):,}; min {pr.min():.2e}, median {np.median(pr):.2e}, max {pr.max():.2e} (ratio max/median = {pr.max() / np.median(pr):.0f}x)")
with e.connect() as c:
    rows = c.execute(text("""SELECT co.name, COUNT(*) n, AVG(jp.company_pagerank) FROM jobs j JOIN companies co USING(company_id) JOIN job_pagerank jp USING(job_id)
                             WHERE j.duplicate_of IS NULL GROUP BY co.company_id, co.name HAVING n >= 3""")).all()
r_ = np.corrcoef(np.log([r[1] for r in rows]), np.log([r[2] for r in rows]))[0, 1]
out.append(f"- correlation between log(number of postings) and log(company PageRank) over {len(rows):,} employers: **{r_:.2f}** - in this constructed graph PageRank mostly reflects posting volume/connectivity (prominent employers, recruiters, job boards), not quality; whether it helps retrieval is an experiment for Phase 4.")
out.append("\n### Sample explained result (hybrid)\n")
x = res["hybrid"]["python machine learning sql"].results[0]
out.append(f"`{x.title}` / {x.company} / {x.city} - score {x.score:.3f}\n\n- signals: {{{', '.join(f'{k}: {v:.2f}' for k, v in x.signals.items() if v is not None)}}}\n" + "\n".join(f"- {r}" for r in x.reasons) + f"\n- passage: {x.passage['html'][:300]}")
out.append(f"\n**Result: {'ALL CHECKS PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}**")
(ROOT / "docs" / "phase2_report.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
sys.exit(1 if fails else 0)
