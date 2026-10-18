"""Scratch launcher for visual checks: the real index and database, search logging off, index and engine warmed.  Not a production server (Flask's built-in server).
    python tools/phase7_scratch/serve_real.py 5057"""
import sys
from jobintel.web import create_app
from jobintel.web.settings import WebSettings

port = int(sys.argv[1]) if len(sys.argv) > 1 else 5057
st = WebSettings.from_env(warmup=False, log_searches=False, secret_key="visual-check-only")
app = create_app(st)
be = app.extensions["jobintel"].backend
be.warm_up(("index", "engine", "catalog", "market"))
app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)
