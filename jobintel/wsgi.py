"""WSGI entrypoint for production deployments (Waitress, Gunicorn, uWSGI).

Usage:
    # Windows / Cross-platform with Waitress:
    python scripts/run_production.py --threads 4 --port 5000

    # Or directly with waitress-serve:
    waitress-serve --listen=127.0.0.1:5000 wsgi:application

    # Linux with Gunicorn:
    gunicorn -w 4 -b 127.0.0.1:5000 "wsgi:application"
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from jobintel.web import create_app
from jobintel.web.settings import WebSettings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("jobintel.wsgi")

settings = WebSettings.from_env()

# Guard against weak secret keys in production
if not settings.testing and settings.secret_key in ("dev-secret-change-me", "test-secret", "dev-secret-change-in-production"):
    log.warning(
        "SECURITY NOTICE: Default SECRET_KEY is in use! Set a strong random SECRET_KEY in your .env file: "
        "python -c 'import secrets; print(secrets.token_hex(32))'"
    )

# Instantiate the Flask application
application = create_app(settings)

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Serving JobIntel WSGI production application on http://{host}:{port} with Waitress")
    serve(application, host=host, port=port, threads=4)
