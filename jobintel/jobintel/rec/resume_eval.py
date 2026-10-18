"""Field-level evaluation of the resume analyser against SYNTHETIC ground truth (see synthetic_resumes.py).  Micro-averaged precision / recall over resumes."""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

import numpy as np


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _prf(pairs: Sequence) -> Dict:
    tp = sum(len(p & t) for p, t in pairs)
    fp = sum(len(p - t) for p, t in pairs)
    fn = sum(len(t - p) for p, t in pairs)
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    F = 2 * P * R / (P + R) if (tp and P == P and R == R and P + R) else (0.0 if tp + fp + fn else float("nan"))
    return {"precision": P, "recall": R, "f1": F, "tp": tp, "fp": fp, "fn": fn}


def field_sets(prof, truth) -> Dict[str, tuple]:
    """(predicted set, true set) per field for one resume."""
    handle = truth.url.split("/")[-1].lower()
    return {
        "email": ({e.lower() for e in prof.emails}, {truth.email.lower()}),
        "phone": ({_digits(p) for p in prof.phones}, {_digits(truth.phone)}),
        "url": ({u.lower().rstrip(".,;)") for u in prof.urls if handle in u.lower()}, {truth.url.lower()}),
        "name": ({prof.name} if prof.name else set(), {truth.name}),
        "organisations": ({o for o in prof.organisations}, set(truth.companies) | set(truth.schools)),
        "companies_only": ({o for o in prof.organisations if o in truth.companies or o not in truth.schools}, set(truth.companies)),
        "skills": ({s["name"].lower() for s in prof.skills}, {s.lower() for s in truth.skills}),
        "skills_listed_only": ({s["name"].lower() for s in prof.skills if s["weight"] == 1.0}, {s.lower() for s in truth.skills}),
        "certifications": ({c.lower() for c in prof.certifications}, {c.lower() for c in truth.certifications}),
    }


def evaluate(profiles: Sequence, truths: Sequence) -> Dict:
    per_field: Dict[str, List] = {}
    exp_err_month, exp_err_year, stated_hits, stated_n = [], [], 0, 0
    sections_ok = 0
    for prof, t in zip(profiles, truths):
        for k, v in field_sets(prof, t).items():
            per_field.setdefault(k, []).append(v)
        if prof.computed_experience_years is not None:
            err = prof.computed_experience_years - t.experience_months / 12.0
            (exp_err_year if any(re.fullmatch(r"\s*(19|20)\d{2}\s*-\s*((19|20)\d{2}|present)\s*", r["text"], re.I) for r in prof.date_ranges if r["counted"]) else exp_err_month).append(err)
        else:
            exp_err_month.append(float("nan"))
        if t.stated_years is not None:
            stated_n += 1
            stated_hits += int(prof.stated_experience_years == float(t.stated_years))
        sections_ok += int({"experience", "education", "skills"} <= set(prof.sections))
    out = {k: _prf(v) for k, v in per_field.items()}
    out["name_exact_match"] = float(np.mean([bool(p) and p == t for p, t in per_field["name"]]))

    def summ(errs):
        a = np.array([e for e in errs if e == e])
        return {"n": int(len(a)), "mean_abs_error_years": float(np.abs(a).mean()) if len(a) else float("nan"), "mean_signed_error_years": float(a.mean()) if len(a) else float("nan"),
                "within_half_year": float((np.abs(a) <= 0.5).mean()) if len(a) else float("nan")}
    out["experience_from_dates"] = {"month_precision_ranges": summ(exp_err_month), "year_only_ranges": summ(exp_err_year)}
    out["experience_unknown_share"] = float(np.mean([p.computed_experience_years is None for p in profiles]))
    out["stated_experience_accuracy"] = {"n": stated_n, "exact": (stated_hits / stated_n) if stated_n else float("nan")}
    out["sections_found_all_three"] = sections_ok / max(len(profiles), 1)
    out["n_resumes"] = len(profiles)
    return out
