"""Names, colours and one-line explanations of the ranking signals.  The SAME colour means the SAME signal on every page (search, recommendations, IR experiments); the labels
are always printed next to the colour, so the colour is never the only carrier of meaning."""
from __future__ import annotations

from typing import Dict, Mapping, Optional

SIGNAL_ORDER = ["ir", "lm", "dfr", "passage", "pagerank", "skill", "skill_match", "content", "knowledge", "dl", "collaborative", "location", "experience", "preference"]

SIGNALS: Dict[str, Dict[str, str]] = {
    "ir": {"label": "Retrieval score", "help": "the hybrid IR score of the job for a query built from your profile (language model + DFR + passage + PageRank + filters)"},
    "lm": {"label": "Language model", "help": "query likelihood: how probable the query is under the job's smoothed word distribution"},
    "dfr": {"label": "DFR", "help": "divergence from randomness (PL2): how far the query words' frequency in the job departs from chance"},
    "passage": {"label": "Best passage", "help": "language-model score of the best-matching passage of the job text"},
    "pagerank": {"label": "PageRank", "help": "link-based authority from the company - career page - job - related-job graph"},
    "skill": {"label": "Skills in query", "help": "share of the skills named in the query that the job lists"},
    "skill_match": {"label": "Skill coverage", "help": "share of the job's skill requirements (idf-weighted) that your profile covers"},
    "content": {"label": "Content", "help": "cosine between your idf-weighted skill vector and the job's"},
    "knowledge": {"label": "Knowledge graph", "help": "coverage of the occupations your skills / role point to in the ESCO + curated knowledge graph"},
    "dl": {"label": "Deep learning", "help": "similarity in the representation spaces learned in Phase 5 (percentile among all jobs of the same space)"},
    "collaborative": {"label": "Collaborative", "help": "item-based collaborative filtering from the jobs you viewed, clicked, saved or applied to"},
    "location": {"label": "Location", "help": "1 when the job is in a city you asked for, 0 when it is elsewhere, neutral when unknown"},
    "experience": {"label": "Experience", "help": "1 when the job's experience band overlaps the one you asked for, less the further apart they are"},
    "preference": {"label": "Preferences", "help": "mean of the preference checks that apply: city, experience, category, employment type"},
}

SEARCH_SIGNALS = ["lm", "dfr", "passage", "skill", "location", "experience", "pagerank"]
REC_SIGNALS = ["ir", "lm", "dfr", "passage", "pagerank", "dl", "content", "collaborative", "knowledge", "skill_match", "preference"]


def contributions(signals: Mapping[str, Optional[float]], weights: Mapping[str, float]) -> Dict[str, float]:
    """weight x value per signal (the parts of the final score).  Signals without a value or weight contribute nothing."""
    out: Dict[str, float] = {}
    for s, v in signals.items():
        w = weights.get(s)
        if v is None or not w:
            continue
        out[s] = float(w) * float(v)
    return out
