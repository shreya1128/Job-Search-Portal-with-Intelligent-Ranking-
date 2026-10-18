#!/usr/bin/env python3
"""Build the retrieval index (kept apart from the database) from the ingested jobs.

   python scripts/build_index.py                  # real datasets only, no duplicates, no fraudulent EMSCAD postings
   python scripts/build_index.py --write-db       # also fill the job_passages and job_terms tables for full-text jobs
Run scripts/compute_pagerank.py first so PageRank is stored in the index too."""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import insert, text
from jobintel.config import DATA_DIR
from jobintel.db.models import JobPassage, JobTerm
from jobintel.db.session import get_engine
from jobintel.ir import passages as PS
from jobintel.ir.index import IndexDoc, InvertedIndex

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=DATA_DIR / "index" / "job_index.pkl")
ap.add_argument("--include-synthetic", action="store_true", help="also index synthetic/crawled-fixture sources (demo only; never for evaluation)")
ap.add_argument("--include-fraud", action="store_true", help="keep EMSCAD postings labelled fraudulent")
ap.add_argument("--write-db", action="store_true")
a = ap.parse_args()
e = get_engine()
t0 = time.time()
where = "j.duplicate_of IS NULL AND j.text_quality <> 'none'"
if not a.include_synthetic:
    where += " AND s.source_type = 'dataset'"
if not a.include_fraud:
    where += " AND j.is_fraudulent = 0"
skills = defaultdict(list)
with e.connect() as c:
    for jid, name in c.execute(text(f"SELECT js.job_id, k.canonical_name FROM job_skills js JOIN skills k USING(skill_id) JOIN jobs j USING(job_id) JOIN crawler_sources s USING(source_id) WHERE {where}")):
        skills[jid].append(name)
    q = text(f"""SELECT j.job_id, s.name, j.title, co.name, j.city, j.cities, j.country, j.experience_min, j.experience_max, j.salary_min, j.salary_max, j.salary_currency,
                 j.salary_period, j.employment_type, j.cluster, j.category, j.text_quality, j.posting_date, j.seniority, j.external_id, j.processed_text
                 FROM jobs j JOIN crawler_sources s USING(source_id) LEFT JOIN companies co USING(company_id) WHERE {where} ORDER BY j.job_id""")
    docs = []
    for r in c.execution_options(stream_results=True).execute(q):
        cities = json.loads(r[5]) if isinstance(r[5], str) else r[5]
        docs.append(IndexDoc(r[0], (r[20] or "").split(), dict(source=r[1], title=r[2], company=r[3], city=r[4], cities=cities, country=r[6], exp_min=r[7], exp_max=r[8],
                                                                  sal_min=r[9], sal_max=r[10], sal_cur=r[11], sal_period=r[12], employment=r[13], cluster=r[14], category=r[15],
                                                                  text_quality=r[16], posting_date=str(r[17]) if r[17] else None, seniority=r[18], external_id=r[19], skills=skills.get(r[0], []))))
    pr = dict(c.execute(text("SELECT job_id, pagerank FROM job_pagerank")).all())
print(f"read {len(docs):,} documents from the database [{time.time() - t0:.0f}s]")
ix = InvertedIndex.build(docs)
if pr:
    floor = min(pr.values())
    ix.pagerank = np.array([pr.get(int(j), floor) for j in ix.job_ids], dtype=np.float64)
else:
    print("WARNING: no PageRank in the database - run scripts/compute_pagerank.py and rebuild")
ix.save(a.out)
print("index:", json.dumps(ix.stats()), f"| file {a.out.stat().st_size / 1e6:.0f} MB | built in {time.time() - t0:.0f}s")

if a.write_db:
    t1 = time.time()
    ids = [int(ix.job_ids[i]) for i in range(ix.n_docs) if ix.meta["text_quality"][i] == "full"]
    with e.begin() as c:
        c.execute(text("DELETE FROM job_terms")); c.execute(text("DELETE FROM job_passages"))
    n_terms = n_pass = 0
    for i0 in range(0, len(ids), 500):
        chunk = ids[i0:i0 + 500]
        with e.connect() as c:
            texts = {j: t for j, t in c.execute(text("SELECT job_id, description_clean FROM jobs WHERE job_id IN :ids").bindparams(__import__("sqlalchemy").bindparam("ids", expanding=True)), {"ids": chunk})}
        prow, trow = [], []
        for j in chunk:
            for p in PS.segment(texts.get(j) or ""):
                prow.append(dict(job_id=j, passage_no=p.no, passage_text=p.text[:60000], n_tokens=p.n_words, section=(p.section or "")[:80] or None))
            toks = ix.doc_tokens(ix.doc_index(j))
            u, cnt = np.unique(toks, return_counts=True)
            trow += [dict(job_id=j, term=ix.terms[t][:100], tf=int(n)) for t, n in zip(u, cnt)]
        with e.begin() as c:
            if prow:
                c.execute(insert(JobPassage).prefix_with("IGNORE", dialect="mysql").prefix_with("OR IGNORE", dialect="sqlite"), prow)
            for k in range(0, len(trow), 20000):
                c.execute(insert(JobTerm).prefix_with("IGNORE", dialect="mysql").prefix_with("OR IGNORE", dialect="sqlite"), trow[k:k + 20000])
        n_terms += len(trow); n_pass += len(prow)
    print(f"wrote {n_pass:,} job_passages and {n_terms:,} job_terms for {len(ids):,} full-text jobs [{time.time() - t1:.0f}s]")
