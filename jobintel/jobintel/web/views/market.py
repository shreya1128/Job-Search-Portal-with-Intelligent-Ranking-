"""Career Insights: aggregates from the searchable job snapshot and the existing occupation graph."""
from __future__ import annotations

import logging
import numpy as np
from flask import Blueprint, render_template, request
from ..backend import Unavailable
from ..forms import Params
from ._common import backend, public_reason

bp = Blueprint("market", __name__)
log = logging.getLogger("jobintel.web")


@bp.route("/career-insights")
def market():
    be = backend()
    p = Params(request.args)
    area = p.text("area", 80)
    role = p.text("role", 120)
    ctx = {"areas": [], "roles": [], "skills": [], "related_roles": [], "area": area, "role": role, "unavailable": None, "coverage": None, "total": None}
    try:
        ix = be.index
        cat = be.catalog
        ctx["total"] = int(ix.n_docs)
        ctx["areas"] = [{"name": n, "count": int(c)} for n, c in be.area_counts() if n and n.lower() not in ("none", "null")][:12]
        ctx["roles"] = [{"name": n.title(), "count": int(c)} for n, c in be.popular_roles(12)]
        docs = np.arange(ix.n_docs, dtype=np.int64)
        if area:
            docs = docs[np.array([str(v).lower() == area.lower() for v in ix.meta["category"]])]
        freqs = cat.market_frequency(docs)
        n_with_skills = int(np.count_nonzero(cat.has_skills[docs])) if len(docs) else 0
        ordered = sorted(freqs.items(), key=lambda kv: (-kv[1], kv[0]))[:18]
        ctx["skills"] = [{"name": n, "count": int(round(fr * n_with_skills)), "share": round(fr * 100, 1)} for n, fr in ordered]
        ctx["coverage"] = len(docs)
        if role:
            kg, why = be.try_get("kg")
            if kg is not None:
                ctx["related_roles"] = kg.match_title(role, top=8)
            elif why:
                ctx["related_unavailable"] = "Occupation links are unavailable right now; market counts are still shown."
    except Unavailable as e:
        ctx["unavailable"] = public_reason(e, "Career Insights")
    except Exception as e:
        log.exception("career insights failed")
        ctx["unavailable"] = "Some market insights could not be loaded. Try again later; the site operator can inspect the server log."
    return render_template("market.html", **ctx)
