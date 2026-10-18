#!/usr/bin/env python3
"""One-off migration for databases created before Phase 4: ir_experiments.ranking_model becomes a free string (it was an ENUM)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import text
from jobintel.db.session import get_engine

e = get_engine()
if e.dialect.name == "mysql":
    with e.begin() as c:
        c.execute(text("ALTER TABLE ir_experiments MODIFY ranking_model VARCHAR(60) NOT NULL"))
    print("ir_experiments.ranking_model is now VARCHAR(60)")
else:
    print("nothing to do for", e.dialect.name)
