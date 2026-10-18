"""PARKED, not registered: placeholder routes for IR / DL experiment pages.  Under the product direction these are not part of the product navigation; the algorithms they would
display run inside search and recommendations.  An optional, quiet "How it works" page may use this module later (stage 6 of the plan).  Nothing here is reachable."""
from flask import Blueprint
bp = Blueprint("experiments", __name__)

@bp.route("/ir-experiments")
def ir():
    return "stub"

@bp.route("/dl-experiments")
def dl():
    return "stub"
