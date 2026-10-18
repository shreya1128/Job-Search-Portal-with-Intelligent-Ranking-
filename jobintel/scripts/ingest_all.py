#!/usr/bin/env python3
"""Load datasets into MySQL.  Datasets that are missing on disk are skipped with a warning (the offline sample always works).

  python scripts/ingest_all.py                      # every real dataset that is available (sample only as fallback)
  python scripts/ingest_all.py --only sample        # offline demo data only
  python scripts/ingest_all.py --only knowledge naukri --limit 500
Order matters and is kept: lexicon -> knowledge base -> job sources -> resumes -> gazetteer."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import DATASET_PATHS as P, GAZETTEER_PATH, LEXICON_PATH, PROCESSED_DIR, ROOT
from jobintel.db.models import Base
from jobintel.db.session import get_engine

ALL = ["sample", "knowledge", "naukri", "emscad", "mcf", "linkedin", "resumes", "gazetteer"]
DEFAULT = [s for s in ALL if s != "sample"]      # synthetic demo jobs are opt-in (--only sample) or an automatic fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=ALL, help="subset of steps (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="max records per source (quick tests)")
    ap.add_argument("--batch", type=int, default=200, help="rows per INSERT batch (lower it if MySQL max_allowed_packet is small)")
    ap.add_argument("--rebuild-lexicon", action="store_true")
    a = ap.parse_args()
    steps = a.only or list(DEFAULT)
    engine = get_engine()
    Base.metadata.create_all(engine)
    log = {}

    from jobintel.extraction.build_lexicon import build
    from jobintel.extraction.lexicon import SkillLexicon
    if a.rebuild_lexicon or not LEXICON_PATH.exists():
        lex, rep = build(P)
        lex.save(LEXICON_PATH)
        print(f"[lexicon] built: {rep.get('entries')} entries ({rep.get('free_text_eligible')} usable in free text)")
    lex = SkillLexicon.load(LEXICON_PATH)

    from jobintel.extraction.extractor import Extractor
    from jobintel.ingest import loaders as L
    from jobintel.ingest import knowledge as K
    from jobintel.ingest.load_db import Ingestor
    ing = Ingestor(engine, Extractor(lex), lex, batch=a.batch)

    def have(path, label):
        ok = Path(path).exists()
        if not ok:
            print(f"[{label}] SKIPPED - file not found: {path}   (set the matching *_CSV / *_DIR variable in .env)")
        return ok

    for step in steps:
        t0 = time.time()
        if step == "sample":
            from jobintel.sample_data import generate, write_jsonl
            if not P["sample"].exists():
                write_jsonl(P["sample"], generate(300))
            log[step] = ing.ingest("sample_synthetic", L.load_jsonl(P["sample"], "sample_synthetic"), limit=a.limit)
        elif step == "knowledge":
            log["kb_curated"] = K.load_curated(engine, lex)
            if have(P["esco_dir"] / "skills_en.csv", "esco"):
                log["kb_esco"] = K.load_esco(engine, P["esco_dir"], lex)
            if have(P["onet_dir"] / "Occupation Data.xlsx", "onet"):
                log["kb_onet"] = K.load_onet(engine, P["onet_dir"], lex)
        elif step == "naukri" and have(P["naukri"], "naukri"):
            log[step] = ing.ingest("naukri", L.load_naukri(P["naukri"]), limit=a.limit)
        elif step == "emscad" and have(P["emscad"], "emscad"):
            log[step] = ing.ingest("emscad", L.load_emscad(P["emscad"]), limit=a.limit)
        elif step == "mcf" and have(P["mcf"], "mcf"):
            log[step] = ing.ingest("mcf", L.load_mcf(P["mcf"]), limit=a.limit)
        elif step == "linkedin":
            src = P["linkedin"] if P["linkedin"].exists() else P["linkedin_meta"]
            if have(src, "linkedin"):
                print(f"[linkedin] using {src.name}" + ("" if "meta" not in src.name else "  (NO job_summary text yet -> skills-only)"))
                log[step] = ing.ingest("linkedin", L.load_linkedin(src), limit=a.limit)
        elif step == "resumes" and have(P["resumes"], "resumes"):
            from jobintel.ingest.resumes import load_resumes
            df = load_resumes(P["resumes"])
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(PROCESSED_DIR / "resumes_clean.csv", index=False)
            log[step] = dict(resumes=len(df), categories=int(df.category.nunique()), median_words=int(df.n_words.median()))
        elif step == "gazetteer":
            from jobintel.ingest.gazetteer import build_gazetteer
            cities = build_gazetteer(engine, GAZETTEER_PATH)
            log[step] = dict(cities=len(cities), sample=cities[:8])
        if step in log or step == "knowledge":
            print(f"[{step}] done in {time.time() - t0:.0f}s ->", json.dumps(log.get(step) or {k: v for k, v in log.items() if k.startswith('kb_')}, default=str)[:600])
    if not a.only and not any(k in log for k in ("naukri", "emscad", "mcf", "linkedin")):
        print("[sample] no real job dataset found -> loading the SYNTHETIC offline sample so the application still works")
        from jobintel.sample_data import generate, write_jsonl
        if not P["sample"].exists():
            write_jsonl(P["sample"], generate(300))
        log["sample"] = ing.ingest("sample_synthetic", L.load_jsonl(P["sample"], "sample_synthetic"))
    from sqlalchemy import text
    with engine.begin() as c:        # housekeeping: companies that no job refers to any more (e.g. after a re-load with new rules)
        n = c.execute(text("DELETE FROM companies WHERE company_id NOT IN (SELECT company_id FROM jobs WHERE company_id IS NOT NULL)")).rowcount
    if n:
        print(f"[housekeeping] removed {n} orphan company rows")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "last_ingest_log.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
