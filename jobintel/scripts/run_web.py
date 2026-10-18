#!/usr/bin/env python3
"""Run the Jobintel web application on this computer (Flask's built-in server).

  python scripts/run_web.py                    # http://127.0.0.1:5000
  python scripts/run_web.py --port 8080

For local use and demonstration only.  It is NOT a production server: for a shared machine put a production WSGI server and HTTPS in front of it (see docs/PHASE7.md, section
"Not done / open") and set SECRET_KEY.  Settings come from the same environment variables / .env as every script (DB_*, SECRET_KEY, JOBINTEL_*; see jobintel/web/settings.py).

Needs the search index (python scripts/build_index.py --write-db) and the loaded database.  A page that needs a missing part shows a plain "not available right now" message and the
reason is written to the console."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.web import create_app                       # noqa: E402
from jobintel.web.settings import WebSettings              # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = only this computer (default); 0.0.0.0 = other computers on the network can connect")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-warmup", action="store_true", help="do not load the search index and models in the background at start-up (they then load on first use)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    overrides = {"warmup": False} if a.no_warmup else {}
    app = create_app(WebSettings.from_env(**overrides))
    if a.host not in ("127.0.0.1", "localhost"):
        logging.getLogger("jobintel.web").warning("listening on %s: anyone who can reach this computer can open the site; this server is not meant for that", a.host)
    # threaded so a slow first search (index loading) does not block other pages; no reloader (it would load the index twice)
    app.run(host=a.host, port=a.port, threaded=True, use_reloader=False, debug=False)


if __name__ == "__main__":
    main()
