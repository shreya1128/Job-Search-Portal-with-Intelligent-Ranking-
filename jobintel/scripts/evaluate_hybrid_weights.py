#!/usr/bin/env python3
"""Evaluate DIFFERENT WEIGHT COMBINATIONS of the 11-signal hybrid on the judged pool of the Phase 4 queries.

    python scripts/evaluate_hybrid_weights.py --judgments relevance_judgments_assistant_labeled.xlsx

Design (read before quoting a number):
 * profile = derived from the QUERY TEXT only (extracted skills, cities, stated experience; the query itself is the preferred role).  There are no real user profiles.
 * candidates = the jobs that were judged for that query (~24 per query).  EVERY configuration re-ranks exactly the same labelled jobs, so no configuration is penalised for
   unjudged documents; the signals are normalised over this pool.  Because the pool was built from IR systems' top-10 lists it favours IR-style signals, and because most pooled
   jobs are relevant (threshold 1) the lenient metrics are compressed - use nDCG and the strict threshold.
 * labels = whatever workbook is given.  With assistant_judged labels the whole evaluation is AI-judged and says so in its output.
 * collaborative filtering is off (no history); the DL signal is on only when the v2 artefacts exist.
 * 20 queries: intervals are wide; paired sign-flip tests with Holm correction decide what is supported."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openpyxl import load_workbook

from jobintel.config import DATA_DIR
from jobintel.db.session import get_engine
from jobintel.eval import metrics as M
from jobintel.eval.judgments import read_labels, resolve_labels
from jobintel.ir.engine import DbDocStore, SearchEngine
from jobintel.ir.index import InvertedIndex
from jobintel.ir.ranker import combine
from jobintel.rec.catalog import ItemCatalog
from jobintel.rec.dlsim import DLSkillSimilarity
from jobintel.rec.hybrid import DEFAULT_REC_WEIGHTS, SIGNALS, HybridRecommender
from jobintel.rec.knowledge import KnowledgeGraph
from jobintel.rec.profile import UserProfile


def profile_from_query(extractor, text: str) -> UserProfile:
    ex = extractor.extract(text)
    exp = ex.get("experience") or {}
    years = float(exp["min"]) if exp.get("explicit") and exp.get("min") is not None else None
    return UserProfile(skills={s["name"]: 1.0 for s in ex["skills"]}, experience_years=years, preferred_role=text, locations=list(ex["locations"]))


def configs(active):
    W = {k: v for k, v in DEFAULT_REC_WEIGHTS.items() if k in active}
    c = {"default hybrid": W}
    for s in active:
        c[f"only {s}"] = {s: 1.0}
    c["IR family only (ir, lm, dfr, passage, pagerank)"] = {s: 1.0 for s in ("ir", "lm", "dfr", "passage", "pagerank") if s in active}
    c["recommender signals only (content, knowledge, skill_match, preference, dl)"] = {s: 1.0 for s in ("content", "knowledge", "skill_match", "preference", "dl") if s in active}
    c["equal weights (all active)"] = {s: 1.0 for s in active}
    for s in active:
        c[f"default without {s}"] = {k: v for k, v in W.items() if k != s}
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--index", default=str(DATA_DIR / "index" / "job_index.pkl"))
    ap.add_argument("--dl-version", default="v2")
    ap.add_argument("--relevance-source", default="assistant_judged")
    ap.add_argument("--out", default=str(DATA_DIR.parent / "docs"))
    a = ap.parse_args()
    db = get_engine()
    ix = InvertedIndex.load(Path(a.index))
    eng = SearchEngine(ix, DbDocStore(db))
    cat = ItemCatalog(ix)
    dl = DLSkillSimilarity.from_files(DATA_DIR, cat, a.dl_version)
    rec = HybridRecommender(eng, cat, KnowledgeGraph.from_db(db), cf=None, dl=dl)
    labels, _ = read_labels(Path(a.judgments))
    by_job, missing = resolve_labels(labels, ix)
    ws = load_workbook(a.judgments, read_only=True)["Queries"]
    rows = [r for r in ws.iter_rows(values_only=True)][1:]
    queries = [(str(r[0]), str(r[1])) for r in rows if r[0]]
    per_q = {}
    active = None
    print(f"DL signal: {'ON (' + dl.label + f', {dl.n_docs_covered:,} indexed jobs have an embedding)' if dl.available else 'OFF - ' + str(dl.reason)}")
    for qid, qtext in queries:
        lab = by_job.get(qid)
        if not lab:
            continue
        prof = profile_from_query(eng.extractor, qtext)
        docs = [ix.doc_index(j) for j in lab]
        resp = rec.recommend(prof, k=len(docs), restrict_docs=docs, collapse_duplicates=False)
        ids = [r.job_id for r in resp.results]
        assert set(ids) == set(lab), (qid, len(ids), len(lab))
        sig = {s: (None if resp.results[0].signals[s] is None else np.array([r.signals[s] for r in resp.results])) for s in SIGNALS}
        per_q[qid] = {"text": qtext, "ids": ids, "sig": sig, "labels": lab, "profile": {"skills": list(prof.skills), "locations": prof.locations, "experience_years": prof.experience_years},
                      "inactive": resp.inactive, "dl_coverage": resp.dl_coverage}
    active = [s for s in SIGNALS if any(q["sig"][s] is not None for q in per_q.values())]            # signals that exist for at least one query; a query drops what it lacks
    n_with = {s: sum(q["sig"][s] is not None for q in per_q.values()) for s in active}
    print(f"{len(per_q)} queries; queries for which each signal exists: {n_with}")
    cfgs = configs(active)
    res = {}
    rng = np.random.default_rng(0)

    nan = lambda x: float("nan") if x is None else float(x)

    def metrics(ranked, lab):                                    # MRR (grade 2) is undefined for a query with no grade-2 job (e.g. q02): NaN, averaged with nanmean
        return (nan(M.ndcg_at_k(ranked, lab, 10)), nan(M.precision_at_k(ranked, lab, 10, 1)), nan(M.precision_at_k(ranked, lab, 10, 2)), nan(M.reciprocal_rank(ranked, lab, 2)))
    tie = {qid: np.random.default_rng(1000 + i).permutation(len(q["ids"])) for i, (qid, q) in enumerate(per_q.items())}      # fixed RANDOM tie-break: pool order is the default ranking, so it must not decide ties
    evidence = {}
    for name, W in cfgs.items():
        vals, ev = [], 0
        for qid, q in per_q.items():
            final, eff = combine({s: q["sig"][s] for s in SIGNALS}, W, len(q["ids"]))
            ev += int(bool(eff))
            order = np.lexsort((tie[qid], -final))
            vals.append(metrics([q["ids"][i] for i in order], q["labels"]))
        res[name] = np.array(vals, dtype=float)
        evidence[name] = ev
    rand = []
    for qid, q in per_q.items():
        v = np.nanmean([metrics([q["ids"][i] for i in rng.permutation(len(q["ids"]))], q["labels"]) for _ in range(300)], axis=0)
        rand.append(v)
    res["random order of the pool"] = np.array(rand)
    evidence["random order of the pool"] = len(per_q)
    base = res["default hybrid"][:, 0]
    comps = {n: M.paired_bootstrap(v[:, 0], base) for n, v in res.items() if n != "default hybrid"}
    adj = dict(zip(comps, M.holm([c["p_value"] for c in comps.values()])))
    md = ["# Phase 6 - weight combinations of the hybrid recommender (judged-pool re-ranking)", "",
          f"**Relevance labels: `{a.relevance_source}`" + (" - these are AI judgments, NOT human labels." if a.relevance_source == "assistant_judged" else "."),
          "**Profiles are derived from the query text (no real users); candidates are the judged jobs of each query; collaborative filtering is off (no history).** "
          + (f"DL signal ON ({dl.label})." if dl.available else f"DL signal OFF ({dl.reason})."), "",
          f"{len(per_q)} queries; signals active for all queries: {', '.join(active)}.  Strict P@10 / MRR use grade 2 as relevant; P@10 (lenient) counts grade >= 1 and is near its ceiling (random order of the pool already scores {np.nanmean(res['random order of the pool'][:, 1]):.2f}).", "",
          "| configuration | queries with evidence | nDCG@10 | P@10 (grade>=1) | P@10 (grade 2) | MRR (grade 2) | nDCG@10 minus default [95% CI] | Holm-adj. p |", "|---|---|---|---|---|---|---|---|"]
    for n, v in sorted(res.items(), key=lambda kv: -np.nanmean(kv[1][:, 0])):
        m = np.nanmean(v, axis=0)
        c = comps.get(n)
        d = f"{c['mean_diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]" if c else "-"
        md.append(f"| {n} | {evidence[n]}/{len(per_q)} | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.3f} | {d} | {adj[n]:.3f} |" if c else f"| {n} | {evidence[n]}/{len(per_q)} | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.3f} | - | - |")
    md += ["", "The p-values compare each configuration with the default over the same queries (paired sign-flip randomisation, Holm-adjusted over all comparisons).  A difference that is not "
           "supported means 'not enough evidence with 20 queries', not 'no effect'.  A configuration whose signals do not exist for a query (e.g. 'only content' for a query with no recognised skill) has no evidence there and ranks that query in a fixed random order; "
           "'queries with evidence' says for how many queries it had any signal.  MRR is averaged over the queries that have at least one grade-2 job.  The default weights are provisional starting values, not claimed optimal.", ""]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "phase6_weight_evaluation.md").write_text("\n".join(md), encoding="utf-8")
    js = {"relevance_source": a.relevance_source, "dl_on": dl.available, "active_signals": active, "queries_with_signal": n_with, "n_queries": len(per_q), "configs": {n: {"mean": np.nanmean(v, axis=0).tolist(), "queries_with_evidence": evidence[n], "per_query_ndcg10": v[:, 0].tolist()} for n, v in res.items()},
          "vs_default_ndcg10": {n: {**c, "holm_p": adj[n]} for n, c in comps.items()}, "profiles": {q: x["profile"] for q, x in per_q.items()}, "unresolved_labels": missing}
    (DATA_DIR / "rec").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "rec" / "weight_evaluation_AI_JUDGED.json").write_text(json.dumps(js, indent=1, default=float))
    print("\n".join(md[6:]))
