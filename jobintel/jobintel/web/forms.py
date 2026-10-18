"""Reading and validating request parameters.  Every user-supplied value goes through one of these helpers: length-limited text, bounded numbers, whitelisted choices.
Problems are collected in ``Params.errors`` (shown to the user next to the form) instead of raising, so a page can still render with the valid part of the input."""
from __future__ import annotations

import re
from typing import Any, List, Mapping, Optional, Sequence

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class Params:
    def __init__(self, source: Mapping[str, Any]):
        self.src = source
        self.errors: List[str] = []

    def has(self, *names: str) -> bool:
        return any((self.src.get(n) or "") != "" for n in names)

    def text(self, name: str, maxlen: int = 200, default: str = "") -> str:
        v = self.src.get(name)
        if v is None:
            return default
        v = _CTRL.sub("", str(v)).strip()
        if len(v) > maxlen:
            self.errors.append(f"{name.replace('_', ' ')} is longer than {maxlen} characters")
            v = v[:maxlen]
        return v or default

    def number(self, name: str, lo: float, hi: float, default: Optional[float] = None, label: Optional[str] = None) -> Optional[float]:
        raw = (self.src.get(name) or "")
        raw = str(raw).strip().replace(",", "")
        if raw == "":
            return default
        try:
            x = float(raw)
        except ValueError:
            self.errors.append(f"{label or name.replace('_', ' ')} must be a number")
            return default
        if x != x or x in (float("inf"), float("-inf")) or not (lo <= x <= hi):
            self.errors.append(f"{label or name.replace('_', ' ')} must be between {lo:g} and {hi:g}")
            return default
        return x

    def integer(self, name: str, lo: int, hi: int, default: Optional[int] = None, label: Optional[str] = None) -> Optional[int]:
        x = self.number(name, lo, hi, None, label)
        return default if x is None else int(x)

    def choice(self, name: str, options: Sequence[Any], default: Any = None) -> Any:
        v = self.src.get(name)
        if v in (None, ""):
            return default
        v = str(v)
        if v not in [str(o) for o in options]:
            self.errors.append(f"{name.replace('_', ' ')} must be one of: {', '.join(str(o) for o in options)}")
            return default
        return next(o for o in options if str(o) == v)

    def flag(self, name: str) -> bool:
        return str(self.src.get(name, "")).strip().lower() in ("1", "true", "on", "yes")

    def csv(self, name: str, max_items: int = 25, maxlen: int = 60) -> List[str]:
        raw = self.src.get(name)
        if raw is None:
            return []
        items = [x.strip() for x in re.split(r"[,\n;]", _CTRL.sub("", str(raw))) if x.strip()]
        if len(items) > max_items:
            self.errors.append(f"{name.replace('_', ' ')}: at most {max_items} entries are used")
            items = items[:max_items]
        return [x[:maxlen] for x in items]

    def many(self, name: str, options: Optional[Sequence[str]] = None) -> List[str]:
        getter = getattr(self.src, "getlist", None)
        vals = getter(name) if getter else ([self.src[name]] if name in self.src else [])
        vals = [str(v) for v in vals if str(v) != ""]
        if options is not None:
            bad = [v for v in vals if v not in options]
            if bad:
                self.errors.append(f"{name.replace('_', ' ')}: unknown value(s) {bad}")
            vals = [v for v in vals if v in options]
        return vals
