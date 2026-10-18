#!/usr/bin/env python3
"""Build data/processed/skill_lexicon.json from the raw datasets (run BEFORE ingest_all)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import DATASET_PATHS, LEXICON_PATH
from jobintel.extraction.build_lexicon import build

lex, report = build(DATASET_PATHS)
lex.save(LEXICON_PATH)
print(json.dumps(report, indent=2))
print("saved", LEXICON_PATH)
