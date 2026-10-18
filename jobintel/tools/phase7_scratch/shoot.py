"""Screenshots of the running site with Chromium (Playwright) at phone, tablet and desktop widths, plus a console / CSP / font report.
    python tools/phase7_scratch/shoot.py BASE_URL OUT_DIR [pages-file]
A page is  name|path|widths   (widths: m=390, t=820, d=1366, comma separated).  Exit code 1 if the browser reported a console error, a CSP violation or a missing font."""
import json, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

WIDTHS = {"m": (390, 844), "t": (820, 1180), "d": (1366, 900)}
base, out = sys.argv[1].rstrip("/"), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
pages = [l.strip().split("|") for l in Path(sys.argv[3]).read_text().splitlines() if l.strip() and not l.startswith("#")]
problems = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
    for name, path, widths in pages:
        for w in widths.split(","):
            vw, vh = WIDTHS[w]
            ctx = browser.new_context(viewport={"width": vw, "height": vh}, device_scale_factor=1)
            pg = ctx.new_page()
            msgs = []
            pg.on("console", lambda m, msgs=msgs: msgs.append((m.type, m.text)) if m.type in ("error", "warning") else None)
            pg.on("pageerror", lambda e, msgs=msgs: msgs.append(("pageerror", str(e))))
            r = pg.goto(base + path, wait_until="networkidle")
            fonts = pg.evaluate("() => ({head: document.fonts.check('600 20px \"Fraunces Variable\"'), body: document.fonts.check('16px \"Atkinson Hyperlegible Next Variable\"')})")
            h_overflow = pg.evaluate("() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1")
            pg.screenshot(path=str(out / f"{name}_{w}.png"), full_page=True)
            status = r.status if r else None
            bad = [m for m in msgs if "favicon" not in m[1].lower()]
            print(f"{name:14s} {w}  status={status}  fonts={fonts}  h-scroll={h_overflow}  console={bad}")
            if bad or h_overflow or not (fonts["head"] and fonts["body"]):
                problems.append((name, w, bad, h_overflow, fonts))
            ctx.close()
    browser.close()
sys.exit(1 if problems else 0)
