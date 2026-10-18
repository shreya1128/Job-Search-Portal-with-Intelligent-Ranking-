#!/usr/bin/env python3
"""MEASURE the resume pipeline on SYNTHETIC resumes: (A) field extraction from clean text, (B) the scanned-resume round trip (render -> degrade -> OCR -> analyse).

    python scripts/evaluate_resume_pipeline.py --n-text 200 --n-ocr 12 --seed 7      # writes data/rec/resume_pipeline_SYNTHETIC.json

All resumes are fictional and generated (jobintel/rec/synthetic_resumes.py); the numbers describe this template's difficulty, NOT accuracy on real resumes or real scans.
OCR variants: clean | rotated 3 deg | gaussian noise | low resolution (50%) | blur + JPEG q40 | two columns | two columns rotated -2 deg."""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import DATA_DIR
from jobintel.extraction.extractor import Extractor
from jobintel.rec.documents import ScannedResumeOCR, char_error_rate
from jobintel.rec.resume import ResumeAnalyzer
from jobintel.rec.resume_eval import evaluate
from jobintel.rec.synthetic_resumes import LABEL, SKILL_POOL, make_resume, render_resume_image

VARIANTS = {"clean": {}, "rotated_3deg": dict(rotate_deg=3.0), "noise_sigma25": dict(noise_sigma=25), "low_res_50pct": dict(scale=0.5), "blur_jpeg40": dict(blur=1.0, jpeg_quality=40),
            "two_columns": dict(columns=2), "two_columns_rot-2": dict(columns=2, rotate_deg=-2.0)}


def show(tag, ev):
    f = lambda k: f"P {ev[k]['precision']:.3f} R {ev[k]['recall']:.3f}"
    e = ev["experience_from_dates"]
    print(f"{tag:20s} name {ev['name_exact_match']:.3f} | email {f('email')} | phone {f('phone')} | org {f('organisations')} | skills {f('skills')} | certs {f('certifications')} | "
          f"exp MAE month-ranges {e['month_precision_ranges']['mean_abs_error_years']:.2f}y (n={e['month_precision_ranges']['n']}) year-only {e['year_only_ranges']['mean_abs_error_years']:.2f}y (n={e['year_only_ranges']['n']})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-text", type=int, default=200)
    ap.add_argument("--n-ocr", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--today", default="2026-10-12")
    a = ap.parse_args()
    today = dt.date.fromisoformat(a.today)
    ex = Extractor()
    pool = [s for s in SKILL_POOL if {x["name"].lower() for x in ex.extract(s, skills_only=True)["skills"]} == {s}]
    an = ResumeAnalyzer(ex)
    rng = np.random.default_rng(a.seed)
    res = {"label": LABEL, "today": a.today, "seed": a.seed, "skill_pool_used": pool, "skill_pool_excluded_not_recognised_by_lexicon": sorted(set(SKILL_POOL) - set(pool))}
    items = [make_resume(rng, today, pool) for _ in range(a.n_text)]
    profs = [an.analyze(t, today=today) for t, _ in items]
    res["A_text"] = evaluate(profs, [t for _, t in items])
    print(f"A. clean text, {a.n_text} synthetic resumes (skill pool {len(pool)} of {len(SKILL_POOL)})")
    show("text", res["A_text"])
    if a.n_ocr == 0:
        out = DATA_DIR / "rec"; out.mkdir(parents=True, exist_ok=True)
        (out / "resume_pipeline_SYNTHETIC.json").write_text(json.dumps(res, indent=1, default=float)); sys.exit(0)
    ocr = ScannedResumeOCR()
    rng = np.random.default_rng(a.seed + 1)
    sub = [make_resume(rng, today, pool) for _ in range(a.n_ocr)]
    res["B_ocr"] = {}
    print(f"\nB. OCR round trip, {a.n_ocr} resumes x {len(VARIANTS)} variants")
    for name, kw in VARIANTS.items():
        t0 = time.time()
        cers, confs, profs, skews = [], [], [], []
        for i, (text, truth) in enumerate(sub):
            img, ref = render_resume_image(text, seed=a.seed * 100 + i, **kw)
            r = ocr.read(img)
            cers.append(char_error_rate(ref, r.text))
            confs.append(r.mean_confidence or 0.0)
            skews.append(r.skew_degrees)
            profs.append(an.analyze(r.text, today=today))
        ev = evaluate(profs, [t for _, t in sub])
        res["B_ocr"][name] = {"cer_mean": float(np.mean(cers)), "cer_max": float(np.max(cers)), "mean_confidence": float(np.mean(confs)), "mean_skew_corrected_deg": float(np.mean(skews)),
                              "seconds_per_page": (time.time() - t0) / len(sub), "fields": ev}
        print(f"  {name:20s} CER mean {np.mean(cers):.4f} max {np.max(cers):.4f} | conf {np.mean(confs):.0f} | skew corrected {np.mean(skews):+.1f} deg | {res['B_ocr'][name]['seconds_per_page']:.1f}s/page")
        show("   fields", ev)
    out = DATA_DIR / "rec"
    out.mkdir(parents=True, exist_ok=True)
    (out / "resume_pipeline_SYNTHETIC.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", out / "resume_pipeline_SYNTHETIC.json")
