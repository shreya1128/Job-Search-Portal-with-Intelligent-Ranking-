#!/usr/bin/env python3
"""Serve the fixture site on http://127.0.0.1:8765 (Ctrl+C to stop) so you can browse it or point a crawler at it."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import FIXTURE_DIR
from jobintel.crawler.fixture_site import serve

if not (FIXTURE_DIR / "index.html").exists():
    sys.exit("Fixture site not built yet: run  python scripts/build_fixture_site.py")
srv, base = serve(FIXTURE_DIR, 8765)
print("serving", base, "- Ctrl+C to stop")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    srv.shutdown()
