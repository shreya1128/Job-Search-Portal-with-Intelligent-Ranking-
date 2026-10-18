"""The Flask web application (Phase 7).  ``create_app()`` builds it; ``scripts/run_web.py`` runs it."""
from .app import create_app

__all__ = ["create_app"]
