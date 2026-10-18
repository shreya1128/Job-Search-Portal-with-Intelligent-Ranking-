#!/usr/bin/env python3
"""Build the link graph from the database, run PageRank, store the scores (job_pagerank) and the constructed edges (job_links).

  graph:  Company -> Career page -> Job -> Related job   (+ Job -> Company)
  observed edges come from the crawler (job_links.is_constructed = 0, kept as they are);
  for the dataset sources, which have no hyperlinks, the edges are CONSTRUCTED (is_constructed = 1):
      company <-> its jobs, and 'related jobs' = same cluster + shared RARE skills.   Run before build_index.py."""
import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import delete, insert, text
from jobintel.db.models import JobLink, JobPageRank
from jobintel.db.session import get_engine
from jobintel.ir.pagerank import LinkGraph, related_job_edges

ap = argparse.ArgumentParser()
ap.add_argument("--damping", type=float, default=0.85)
ap.add_argument("--related-k", type=int, default=3)
ap.add_argument("--max-skill-df", type=int, default=200)
ap.add_argument("--min-shared-idf", type=float, default=8.0)
a = ap.parse_args()
e = get_engine()
t0 = time.time()
with e.connect() as c:
    jobs = c.execute(text("SELECT j.job_id, j.company_id, j.cluster, s.source_type FROM jobs j JOIN crawler_sources s USING(source_id) WHERE j.duplicate_of IS NULL")).all()
    comp_name = dict(c.execute(text("SELECT company_id, name FROM companies")).all())
    pairs = c.execute(text("""SELECT js.job_id, js.skill_id FROM job_skills js JOIN jobs j USING(job_id) JOIN crawler_sources s USING(source_id) JOIN skills k USING(skill_id)
                              WHERE j.duplicate_of IS NULL AND s.source_type='dataset' AND k.skill_type NOT IN ('soft','education')""")).all()
    observed = c.execute(text("SELECT link_type, src_company_id, src_job_id, dst_company_id, dst_job_id, weight FROM job_links WHERE is_constructed = 0")).all()
print(f"jobs in graph: {len(jobs):,}   job-skill pairs: {len(pairs):,}   observed (crawled) edges: {len(observed):,}")

# ---- constructed 'related job' edges (dataset sources only) -------------------------------------------------------------
ds = [(j[0], j[1], j[2]) for j in jobs if j[3] == "dataset"]
row_of = {jid: i for i, (jid, _, _) in enumerate(ds)}
skill_ids = sorted({p[1] for p in pairs})
col_of = {s: i for i, s in enumerate(skill_ids)}
rows = np.array([row_of[j] for j, _ in pairs if j in row_of]); cols = np.array([col_of[s] for j, s in pairs if j in row_of])
clusters = {c: i for i, c in enumerate(sorted({d[2] or "" for d in ds}))}
cl = np.array([clusters[d[2] or ""] for d in ds])
rs, rd, rw = related_job_edges(rows, cols, len(ds), len(skill_ids), cl, k=a.related_k, max_df=a.max_skill_df, min_shared_idf=a.min_shared_idf)
print(f"related-job edges: {len(rs):,}  (jobs with >=1 related job: {len(set(rs.tolist())):,} of {len(ds):,})  [{time.time() - t0:.0f}s]")

# ---- graph ----------------------------------------------------------------------------------------------------------------
g = LinkGraph()
career_done = set()


def company_career(cid):
    if cid not in career_done:
        career_done.add(cid)
        g.edge(("company", cid), ("career", cid))


for jid, cid, _ in ds:
    g.node("job", jid)
    if cid:
        company_career(cid)
        g.edge(("career", cid), ("job", jid))
        g.edge(("job", jid), ("company", cid))
for s_, d_, w_ in zip(rs, rd, rw):
    g.edge(("job", ds[s_][0]), ("job", ds[d_][0]), float(w_))
for lt, sc, sj, dc, dj, w_ in observed:
    if lt == "company_to_job":
        company_career(sc); g.edge(("career", sc), ("job", dj), w_)
    elif lt == "job_to_company":
        company_career(dc); g.edge(("job", sj), ("company", dc), w_)
    else:
        g.edge(("job", sj), ("job", dj), w_)
for jid, cid, _, st in jobs:                      # make sure every job exists as a node (jobs without any link are dangling nodes)
    g.node("job", jid)
pr = g.run(damping=a.damping)
res = g.last
print(f"graph: {g.n_nodes:,} nodes, {len(g.src):,} edges | PageRank converged={res.converged} in {res.iterations} iterations (delta {res.delta:.2e})")

# ---- store ----------------------------------------------------------------------------------------------------------------
version = dt.datetime.now().strftime("v1-%Y%m%d-%H%M")
rows_pr = [dict(job_id=jid, pagerank=pr[("job", jid)], company_pagerank=pr.get(("company", cid)) if cid else None, graph_version=version) for jid, cid, _, _ in jobs]
full = {"src_company_id": None, "src_job_id": None, "dst_company_id": None, "dst_job_id": None}
links = []
for jid, cid, _ in ds:
    if cid:
        links.append({**full, "link_type": "company_to_job", "src_company_id": cid, "dst_job_id": jid, "is_constructed": True, "weight": 1.0})
        links.append({**full, "link_type": "job_to_company", "src_job_id": jid, "dst_company_id": cid, "is_constructed": True, "weight": 1.0})
links += [{**full, "link_type": "related_job", "src_job_id": ds[s_][0], "dst_job_id": ds[d_][0], "is_constructed": True, "weight": float(w_)} for s_, d_, w_ in zip(rs, rd, rw)]
with e.begin() as c:
    c.execute(delete(JobPageRank))
    c.execute(delete(JobLink).where(JobLink.is_constructed == True))     # noqa: E712  (crawler-observed edges stay)
    for i in range(0, len(rows_pr), 5000):
        c.execute(insert(JobPageRank), rows_pr[i:i + 5000])
    for i in range(0, len(links), 5000):
        c.execute(insert(JobLink), links[i:i + 5000])
print(f"stored {len(rows_pr):,} job_pagerank rows and {len(links):,} constructed job_links  (graph_version {version})  [{time.time() - t0:.0f}s total]")

# ---- sanity report ------------------------------------------------------------------------------------------------------------
jp = np.array([pr[("job", j[0])] for j in jobs])
print(f"job PageRank: min {jp.min():.2e}  median {np.median(jp):.2e}  max {jp.max():.2e}  (uniform would be {1 / g.n_nodes:.2e})")
top = sorted(((v, k[1]) for k, v in pr.items() if k[0] == "company"), reverse=True)[:8]
print("top companies by PageRank:", [(comp_name.get(i, i)[:28], f"{v:.1e}") for v, i in top])
