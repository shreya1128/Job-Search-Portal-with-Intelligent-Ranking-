"""Resume corpus loader (Kaggle 'Resume Dataset': Resume.csv with 2,484 resumes, 24 categories).
Cleans encoding debris and builds a title-masked text (the resume usually starts with its job title, which would
leak the category label in a classification experiment)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from jobintel.processing.text import fix_mojibake
import unicodedata


def clean_resume_text(s: str) -> str:
    s = fix_mojibake(str(s or ""))
    s = unicodedata.normalize("NFKC", s).replace("\xa0", " ")
    return re.sub(r"[ \t]{2,}", "  ", s).strip()


_CAPS_TITLE = re.compile(r"^(?:[A-Z0-9][A-Z0-9 &/,\-\.\(\)']{2,}?)(?:\s{2,}|$)")


def mask_title(text: str) -> str:
    """Drop the leading ALL-CAPS title block(s) ('HR ADMINISTRATOR/MARKETING ASSOCIATE   HR ADMINISTRATOR   Summary ...')."""
    t = text.lstrip()
    for _ in range(3):
        m = _CAPS_TITLE.match(t)
        if not m or len(m.group(0)) > 120:
            break
        t = t[m.end():].lstrip()
    return t


def load_resumes(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"ID": "resume_id", "Category": "category"})
    df["text_clean"] = df["Resume_str"].map(clean_resume_text)
    df["text_masked"] = df["text_clean"].map(mask_title)
    df["n_words"] = df["text_masked"].str.split().str.len()
    return df[["resume_id", "category", "text_clean", "text_masked", "n_words"]]
