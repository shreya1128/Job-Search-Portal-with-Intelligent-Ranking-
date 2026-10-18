#!/usr/bin/env python3
"""Analyse a resume (text, .docx, .pdf - text layer or scanned -, or an image/scan), recommend jobs and show the skill gap.

    python scripts/analyze_resume.py --file data/samples/sample_resume.txt --role "data analyst" --city Pune --k 8 --gap 2
    python scripts/analyze_resume.py --file cv.txt --json out.json --save-user alice      # --save-user writes profile + recommendations + gaps to the DB

Needs the built index (scripts/build_index.py) and the database.  Collaborative filtering is OFF here (a new user has no history); the DL signal is OFF until Phase 5 artefacts exist."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy.orm import Session

from jobintel.config import DATA_DIR
from jobintel.db.session import get_engine
from jobintel.extraction.extractor import Extractor
from jobintel.ir.engine import DbDocStore, SearchEngine
from jobintel.ir.index import InvertedIndex
from jobintel.rec import store
from jobintel.rec.catalog import ItemCatalog
from jobintel.rec.dlsim import DLSkillSimilarity
from jobintel.rec.documents import analyze_document
from jobintel.rec.hybrid import HybridRecommender
from jobintel.rec.knowledge import KnowledgeGraph
from jobintel.rec.profile import UserProfile
from jobintel.rec.resume import ResumeAnalyzer
from jobintel.rec.skillgap import analyze_gap

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--role")
    ap.add_argument("--city", action="append")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--gap", type=int, default=2, help="skill-gap report for the top N recommendations")
    ap.add_argument("--index", default=str(DATA_DIR / "index" / "job_index.pkl"))
    ap.add_argument("--strict-city", action="store_true", help="only recommend jobs in the given cities")
    ap.add_argument("--json")
    ap.add_argument("--save-user")
    a = ap.parse_args()

    db = get_engine()
    ix = InvertedIndex.load(Path(a.index))
    eng = SearchEngine(ix, DbDocStore(db))
    cat = ItemCatalog(ix)
    kg = KnowledgeGraph.from_db(db)
    r, doc = analyze_document(Path(a.file), ResumeAnalyzer(eng.extractor))
    text = doc.text
    print(f"== RESUME  {a.file}  ({r.n_words} words; read as {doc.method}" + (f", OCR confidence {doc.ocr_confidence:.0f}" if doc.ocr_confidence is not None else "") + ")")
    print(f"name (heuristic): {r.name} | emails: {r.emails} | phones: {r.phones} | urls: {r.urls}")
    print(f"sections: {sorted(r.sections)} | organisations (heuristic): {r.organisations}")
    print(f"experience: {r.experience_years} y ({r.experience_source}); stated {r.stated_experience_years}, from date ranges {r.computed_experience_years}")
    print(f"education: {r.best_education} | certifications: {r.certifications} | roles: {r.roles} | locations: {r.locations}")
    print("skills: " + ", ".join(f"{s['name']}({s['weight']:.1f})" for s in r.skills))
    for w in r.warnings:
        print("  warning:", w)
    prof = UserProfile.from_resume(r, preferred_role=a.role, locations=a.city)
    dl = DLSkillSimilarity.from_files(DATA_DIR, cat, "v2")
    rec = HybridRecommender(eng, cat, kg, cf=None, dl=dl)
    resp = rec.recommend(prof, k=a.k, strict_location=a.strict_city)
    print(f"\n== RECOMMENDATIONS  (IR query: {resp.ir_query!r}; {resp.n_candidates} candidates)")
    print("active weights:", {k: round(v, 3) for k, v in resp.weights_effective.items()})
    print("INACTIVE:", resp.inactive)
    if resp.dl_coverage is not None:
        print(f"DL similarity: ON ({dl.label}); {resp.dl_coverage:.0%} of the candidates have an embedding (the others are scored neutrally)")
    for w in resp.warnings:
        print("  warning:", w)
    for x in resp.results:
        print(f"{x.rank:2d}. {x.final_score:.3f}  {x.title[:48]:48s} | {str(x.company)[:22]:22s} | {x.city} | {x.source}")
        print("      " + " | ".join(x.reasons[1:3]))
    gaps = []
    for x in resp.results[: a.gap]:
        di = ix.doc_index(x.job_id)
        g = analyze_gap(cat, kg, prof.skills, job_di=di)
        gaps.append((x.job_id, g))
        print(f"\n== SKILL GAP for {g.target}  (coverage {g.skill_coverage}, {g.n_similar_jobs} similar jobs, occupation: {g.occupation})")
        for row in g.missing[:6]:
            print(f"   missing {row.skill:28s} importance {row.importance:.2f}  market {row.market_frequency:.0%}" + (f"  (you have related: {row.related_held})" if row.related_held else ""))
    if a.json:
        Path(a.json).write_text(json.dumps({"resume": r.to_dict(), "recommendations": resp.to_dict(), "gaps": [g.to_dict() for _, g in gaps]}, indent=1, default=str), encoding="utf-8")
    if a.save_user:
        with Session(db) as s:
            uid = store.create_user(s, a.save_user, f"{a.save_user}@example.invalid")
            print("\nsaved profile:", store.save_profile(s, uid, prof, r, resume_text=text, filename=Path(a.file).name))
            print("saved recommendations:", store.save_recommendations(s, uid, resp))
            for jid, g in gaps:
                store.save_skill_gaps(s, uid, jid, g)
            s.commit()
