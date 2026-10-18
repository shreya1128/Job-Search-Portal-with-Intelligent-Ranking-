"""Build the data-driven skill lexicon: curated seed + frequent dataset tags + O*NET hot technologies + ESCO digital skills.

Free-text eligibility rules (Entry.ambiguous = False means 'may be matched inside descriptions'):
  * seed entries: eligible (single letters excluded);
  * multi-word tags (2-4 tokens) seen >= min_df times: eligible;
  * single-word dataset tags: eligible only if corroborated by O*NET/ESCO or containing a digit/symbol (c#, s3, node.js);
    frequency alone is NOT enough ('retail', 'strategy', 'food' are frequent tags but not skills in running text);
  * everything else is kept ONLY for normalising dataset tags (ambiguous=True).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from jobintel.extraction.lexicon import SkillLexicon, normalize_tag
from jobintel.extraction.seed_skills import FREE_TEXT_BLOCKLIST, GENERIC_SINGLE_WORDS


def _tokens(s: str) -> int:
    return len(re.findall(r"[a-z0-9+#.]+", s))


def build(paths: Dict[str, Path], min_df: Dict[str, int] | None = None) -> tuple[SkillLexicon, dict]:
    from jobintel.ingest import loaders as L
    min_df = min_df or {"naukri": 8, "mcf": 25, "linkedin": 15}
    lex = SkillLexicon.from_seed()
    report: dict = {"seed": len(lex)}
    counts: Dict[str, Counter] = {}
    srcs = {"naukri": (L.load_naukri, paths["naukri"]), "mcf": (L.load_mcf, paths["mcf"]), "linkedin": (L.load_linkedin, None)}
    for name, (fn, p) in srcs.items():
        if name == "linkedin":
            p = next((q for q in (paths.get("linkedin"), paths.get("linkedin_meta")) if q and Path(q).exists()), None)
        if not p or not Path(p).exists():
            report[f"{name}_missing"] = True
            continue
        c: Counter = Counter()
        for rec in fn(p):
            kinds = rec.extras.get("tag_kinds", {})
            for t in rec.skills:
                if kinds.get(t, "skill") != "skill":
                    continue
                n = lex.normalize(t)
                if n:
                    c[n] += 1
        counts[name] = c
        report[f"{name}_distinct_tags"] = len(c)

    corroborated: set = set()
    # ---- O*NET hot technologies ------------------------------------------------------------------------------------
    onet = Path(paths["onet_dir"]) / "Software Skills.xlsx"
    if onet.exists():
        sw = pd.read_excel(onet)
        hot = sw[(sw["Hot Technology"] == "Y") | (sw["In Demand"] == "Y")]["Workplace Example"].dropna().astype(str)
        for ex in set(hot):
            ex = re.sub(r"\s+(software|platform|tool|tools|application)$", "", ex.strip(), flags=re.I)
            n = lex.normalize(ex)
            if n and len(n) <= 40 and _tokens(n) <= 4:
                lex.add(n, "tool", "onet", df=0)
                corroborated.add(n)
        report["onet_hot_added"] = len(corroborated)
    # ---- ESCO digital skills -----------------------------------------------------------------------------------------
    esco = Path(paths["esco_dir"]) / "digitalSkillsCollection_en.csv"
    n_esco = 0
    if esco.exists():
        dg = pd.read_csv(esco, usecols=["preferredLabel", "altLabels"])
        for r in dg.itertuples(index=False):
            for lab in [str(r.preferredLabel)] + [a for a in str(r.altLabels).split("\n") if a and a != "nan"][:4]:
                n = lex.normalize(re.sub(r"\(.*?\)", "", lab))
                if n and 1 <= _tokens(n) <= 3 and len(n) <= 40:
                    lex.add(n, "technical", "esco")
                    corroborated.add(n)
                    n_esco += 1
        report["esco_digital_added"] = n_esco
    # ---- dataset tags -------------------------------------------------------------------------------------------------
    total = Counter()
    for name, c in counts.items():
        for canon, df in c.items():
            if df >= min_df[name]:
                typ = "technical" if (name == "naukri" and canon in corroborated) else "other"   # 'technical' must be vetted, not assumed
                e = lex.add(canon, typ, name, df)
                total[canon] += df
    for canon, e in lex.entries.items():
        e.df = max(e.df, total.get(canon, 0))
        if canon in FREE_TEXT_BLOCKLIST:
            e.ambiguous = True
            continue
        if "seed" in e.sources:
            e.ambiguous = len(canon) == 1
            continue
        toks = _tokens(canon)
        if canon in GENERIC_SINGLE_WORDS:
            e.ambiguous = True
        elif toks >= 2:
            e.ambiguous = False
        else:  # single token
            e.ambiguous = not (canon in corroborated or any(ch in canon for ch in "+#.0123456789"))
    report["entries"] = len(lex)
    report["free_text_eligible"] = sum(1 for e in lex.entries.values() if not e.ambiguous)
    report["by_type"] = dict(Counter(e.type for e in lex.entries.values()))
    return lex, report
