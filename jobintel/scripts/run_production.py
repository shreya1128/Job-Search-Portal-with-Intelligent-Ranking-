#!/usr/bin/env python3
"""Run JobIntel in production mode with Waitress WSGI server.

Usage:
    python scripts/run_production.py [--host 127.0.0.1] [--port 5000] [--threads 4]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from waitress import serve
from wsgi import application


def main() -> None:
    ap = argparse.ArgumentParser(description="Run JobIntel production WSGI server")
    ap.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    ap.add_argument("--threads", type=int, default=4, help="Number of worker threads (default: 4)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("jobintel.production")
    log.info("Starting JobIntel production server with Waitress on http://%s:%d (threads=%d)", args.host, args.port, args.threads)
    serve(application, host=args.host, port=args.port, threads=args.threads)


if __name__ == "__main__":
    main()
