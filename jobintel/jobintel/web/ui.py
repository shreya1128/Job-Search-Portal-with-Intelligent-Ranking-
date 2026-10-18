"""Small Jinja filters shared by all templates."""
from __future__ import annotations

import datetime as dt
import math
import re
from typing import Any, Optional

from markupsafe import Markup, escape

_URL = re.compile(r"(https?://[^\s<>\"']+)")
_DERIVED_EMPLOYER = re.compile(r"^Company-[0-9a-f]{6}$")

# Job areas.  The `cluster` of a job is a title-keyword grouping made when the data was loaded (common.cluster_for_title), not a label a person assigned, so the
# pages call them "job areas" and say they are grouped by title.  `other` is everything the keywords did not catch.
AREAS = {
    "data_ai": "Data and AI", "software_it": "Software and IT", "finance": "Finance and accounting", "sales_marketing": "Sales and marketing", "healthcare": "Healthcare",
    "hr_admin": "HR and administration", "engineering_manufacturing": "Engineering and manufacturing", "education": "Education and training",
    "retail_hospitality": "Retail and hospitality", "logistics_transport": "Logistics and transport", "legal_public": "Legal and public sector", "other": "Other roles",
}
SOURCES = {"emscad": "EMSCAD dataset", "linkedin": "LinkedIn dataset", "naukri": "Naukri dataset", "mycareersfuture": "MyCareersFuture dataset"}


def pct(x: Optional[float], digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * float(x):.{digits}f}%"


def fnum(x: Any, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{float(x):,.{digits}f}"


def intc(x: Any) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return "n/a"


def nice_dt(x: Any) -> str:
    if isinstance(x, (dt.datetime, dt.date)):
        return x.strftime("%d %b %Y")
    return str(x or "")


def paragraphs(text: Optional[str]) -> Markup:
    """Plain text -> escaped HTML paragraphs (blank line = new paragraph, single newline = line break); bare links are NOT made clickable (job text is untrusted)."""
    if not text:
        return Markup("")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", str(text)) if b.strip()]
    return Markup("".join("<p>" + "<br>".join(str(escape(line)) for line in b.splitlines()) + "</p>" for b in blocks))


def employer(name: Optional[str]) -> Optional[str]:
    """The employer's name, or None when the data only has a generated placeholder such as 'Company-4c8196' (the EMSCAD data is anonymised)."""
    if not name or _DERIVED_EMPLOYER.match(str(name).strip()):
        return None
    return str(name).strip()


def safe_url(url: Optional[str]) -> Optional[str]:
    """An http(s) link from the data, or None.  Anything else (javascript:, data:, relative paths, control characters) is never put into a page."""
    if not url:
        return None
    u = str(url).strip()
    if len(u) > 2000 or re.search(r"[\x00-\x20<>\"']", u):
        return None
    return u if re.match(r"^https?://[^/\s]+", u, re.I) else None


def area_label(key: Optional[str]) -> str:
    return AREAS.get(key or "", (key or "").replace("_", " ").capitalize())


def source_label(key: Optional[str]) -> str:
    return SOURCES.get((key or "").lower(), (key or "").capitalize())


def init_ui(app) -> None:
    app.jinja_env.filters.update(pct=pct, fnum=fnum, intc=intc, nice_dt=nice_dt, paragraphs=paragraphs, employer=employer, safe_url=safe_url, area_label=area_label, source_label=source_label)
    app.jinja_env.globals.update(AREAS=AREAS)
