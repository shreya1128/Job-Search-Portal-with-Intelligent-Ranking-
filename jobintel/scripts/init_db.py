#!/usr/bin/env python3
"""Create (or recreate) all tables.   python scripts/init_db.py [--drop] [--export-docs]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import ROOT
from jobintel.db.export_schema import export
from jobintel.db.models import ALL_TABLES, Base
from jobintel.db.session import get_engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", action="store_true", help="drop all tables first (DESTROYS DATA)")
    ap.add_argument("--export-docs", action="store_true", help="write docs/schema.sql and docs/er_diagram.mmd")
    a = ap.parse_args()
    engine = get_engine()
    print("database:", engine.url.render_as_string(hide_password=True))
    if a.drop:
        Base.metadata.drop_all(engine)
        print("dropped existing tables")
    Base.metadata.create_all(engine)
    print(f"created {len(ALL_TABLES)} tables:", ", ".join(ALL_TABLES))
    if a.export_docs:
        export(ROOT / "docs")
        print("wrote docs/schema.sql and docs/er_diagram.mmd")


if __name__ == "__main__":
    main()
