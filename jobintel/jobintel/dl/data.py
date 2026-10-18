"""Classification dataset for the deep-learning chapter.

TASK     predict a job's FUNCTION (Information Technology, Sales, Engineering, Customer Service, ...) from the TEXT of its description.
LABELS   the `function` field supplied by the EMSCAD dataset (a real label chosen by the advertiser; the title-keyword 'cluster' heuristic is NOT used).
INPUT    description + requirements + benefits text only - the TITLE is deliberately left out (it is the strongest shortcut to the label).
HONEST SPLIT
   1. near-duplicate descriptions (SimHash distance <= 3, ANY title/place) are collapsed to one document - 15% of EMSCAD's descriptions repeat an
      earlier one and 91% of those pairs carry the same label, so a random split would let the model memorise test jobs;
   2. documents are split BY EMPLOYER (stratified by class): an employer's template text never appears in two splits.
Rare classes (< min_per_class after collapsing) and the vague class 'Other' are dropped; the list is stored with the dataset."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sqlalchemy import text

from jobintel.processing.dedupe import NearDuplicateIndex, simhash
from jobintel.processing.text import index_tokens

SQL = """SELECT j.job_id, j.company_id, j.category AS label, j.description_clean AS text FROM jobs j JOIN crawler_sources s USING(source_id)
         WHERE s.name = 'emscad' AND j.is_fraudulent = 0 AND j.text_quality = 'full' AND j.category IS NOT NULL AND j.duplicate_of IS NULL"""


def build_classification_dataset(engine, min_per_class: int = 250, seed: int = 42, min_words: int = 30) -> (pd.DataFrame, Dict):
    df = pd.read_sql(text(SQL), engine)
    info: Dict = {"raw_labelled_jobs": len(df)}
    df = df[df.text.str.split().str.len() >= min_words].reset_index(drop=True)
    ix = NearDuplicateIndex()
    keep = []
    for i, t in enumerate(df.text):
        h = simhash(index_tokens(t))
        keep.append(h is None or ix.add_or_match(i, h) is None)
    info["near_duplicates_removed"] = int(len(df) - sum(keep))
    df = df[np.array(keep)].reset_index(drop=True)
    df = df[df.label != "Other"]
    counts = df.label.value_counts()
    classes = sorted(counts[counts >= min_per_class].index)
    df = df[df.label.isin(classes)].reset_index(drop=True)
    info.update(classes=classes, class_counts={c: int((df.label == c).sum()) for c in classes}, jobs=len(df))
    df["group"] = np.where(df.company_id.notna(), df.company_id.fillna(-1).astype(int), -df.job_id)      # jobs without employer are their own group
    df["split"] = "train"
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    folds = [te for _, te in sgkf.split(df.text, df.label, df.group)]
    df.loc[folds[0], "split"] = "test"
    df.loc[folds[1], "split"] = "val"
    g = df.groupby("group").split.nunique()
    info["employer_groups_in_more_than_one_split"] = int((g > 1).sum())
    info["split_sizes"] = df.split.value_counts().to_dict()
    info["split_class_share"] = {s: {c: round(float((df[df.split == s].label == c).mean()), 3) for c in classes} for s in ("train", "val", "test")}
    return df, info


def save_dataset(df: pd.DataFrame, info: Dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(directory / "classification.csv", index=False)
    (directory / "classification_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


def load_dataset(directory: Path):
    return pd.read_csv(directory / "classification.csv"), json.loads((directory / "classification_info.json").read_text(encoding="utf-8"))
