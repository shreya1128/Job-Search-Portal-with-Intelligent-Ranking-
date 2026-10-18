"""WCAG 2.x contrast check for the colour pairs the stylesheet uses.  Run:  python tools/phase7_scratch/contrast.py   (exit code 1 if a pair fails its target).

Semi-transparent tints (e.g. rgba amber on the page) are composited over the colour underneath before measuring."""
import sys

def hx(s):
    s = s.lstrip("#"); return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))

def over(fg_rgb, alpha, bg_rgb):
    return tuple(round(alpha * f + (1 - alpha) * b) for f, b in zip(fg_rgb, bg_rgb))

def lum(c):
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(v) for v in c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def ratio(a, b):
    la, lb = lum(a), lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)

BG, SURF, SURF2, SURF3 = hx("121110"), hx("1a1917"), hx("22201d"), hx("2a2723")
TEXT, MUTED, FAINT = hx("f3efe8"), hx("a8a298"), hx("8a847a")
AMBER, AMBER_INK = hx("d9a441"), hx("1a1405")
SAGE, BRICK, LINK = hx("9bbd97"), hx("dc9082"), hx("e6b95e")
AMBER_SOFT_ON_SURF = over(hx("d9a441"), 0.14, SURF)
SAGE_SOFT_ON_SURF = over(hx("8fae8b"), 0.14, SURF)
BRICK_SOFT_ON_SURF = over(hx("c9786a"), 0.13, SURF)
MARK_ON_SURF = over(hx("d9a441"), 0.24, SURF)

PAIRS = [  # (label, fg, bg, target)
    ("body text on page", TEXT, BG, 4.5), ("body text on card", TEXT, SURF, 4.5), ("body text on raised", TEXT, SURF3, 4.5),
    ("muted text on page", MUTED, BG, 4.5), ("muted text on card", MUTED, SURF, 4.5), ("muted text on raised", MUTED, SURF3, 4.5),
    ("faint (placeholder, disabled) on page", FAINT, BG, 4.5), ("faint on card", FAINT, SURF, 4.5),
    ("link on page", LINK, BG, 4.5), ("link on card", LINK, SURF, 4.5),
    ("amber on page (focus ring, accents)", AMBER, BG, 3.0), ("amber ring on card", AMBER, SURF, 3.0),
    ("amber button text", AMBER_INK, AMBER, 4.5), ("pager active text", AMBER_INK, AMBER, 4.5),
    ("sage chip text", SAGE, SAGE_SOFT_ON_SURF, 4.5), ("brick chip text on page", BRICK, BG, 4.5), ("brick chip text on card", BRICK, SURF, 4.5), ("brick on brick tint", BRICK, BRICK_SOFT_ON_SURF, 4.5),
    ("amber chip text", hx("f2cd85"), AMBER_SOFT_ON_SURF, 4.5), ("text on highlight", TEXT, MARK_ON_SURF, 4.5),
    ("chip text on chip", hx("d9d4ca"), SURF2, 4.5), ("excerpt text on card", hx("cfc9bf"), SURF, 4.5), ("job body text on page", hx("e2ddd3"), BG, 4.5),
    ("form-field border vs page (UI component, 3:1)", hx("6f695f"), BG, 3.0), ("form-field border vs card", hx("6f695f"), SURF, 3.0), ("checkbox border vs card", hx("6f695f"), SURF, 3.0),
]
bad = 0
for label, fg, bg, target in PAIRS:
    r = ratio(fg, bg)
    ok = r >= target
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {r:5.2f}:1  (need {target})  {label}")
sys.exit(1 if bad else 0)
