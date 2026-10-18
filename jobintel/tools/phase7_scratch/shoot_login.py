"""Screenshots of the logged-in pages with Chromium (Playwright): logs in through the real login form, visits each page at phone, tablet and desktop widths, and reports console
errors, CSP violations, missing fonts and horizontal scrolling.   python tools/phase7_scratch/shoot_login.py BASE_URL OUT_DIR USER PASSWORD
Also takes two screenshots of states that need an interaction: the Profile page after an invalid entry, and Account settings after a wrong password."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

WIDTHS = {"m": (390, 844), "t": (820, 1180), "d": (1366, 900)}
base, out, user, pw = sys.argv[1].rstrip("/"), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
out.mkdir(parents=True, exist_ok=True)
PAGES = [("saved", "/saved", "d,t,m"), ("saved_applied", "/saved?status=applied", "d"), ("profile", "/profile", "d,t,m"), ("workspace", "/dashboard", "d,t,m"), ("account", "/account", "d,m")]
problems = []


def report(name, w, pg, msgs, status=None):
    fonts = pg.evaluate("() => ({head: document.fonts.check('600 20px \"Fraunces Variable\"'), body: document.fonts.check('16px \"Atkinson Hyperlegible Next Variable\"')})")
    h = pg.evaluate("() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1")
    bad = [m for m in msgs if "favicon" not in m[1].lower()]
    print(f"{name:20s} {w}  status={status}  fonts={fonts}  h-scroll={h}  console={bad}")
    if bad or h or not (fonts["head"] and fonts["body"]):
        problems.append((name, w))


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
    for w, (vw, vh) in WIDTHS.items():
        ctx = browser.new_context(viewport={"width": vw, "height": vh}, device_scale_factor=1)
        pg = ctx.new_page()
        msgs = []
        pg.on("console", lambda m, msgs=msgs: msgs.append((m.type, m.text)) if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e, msgs=msgs: msgs.append(("pageerror", str(e))))
        pg.goto(base + "/login", wait_until="networkidle")
        pg.fill("#identifier", user)
        pg.fill("#password", pw)
        pg.click("button[type=submit]")
        pg.wait_for_load_state("networkidle")
        for name, path, widths in PAGES:
            if w not in widths.split(","):
                continue
            msgs.clear()
            r = pg.goto(base + path, wait_until="networkidle")
            pg.screenshot(path=str(out / f"{name}_{w}.png"), full_page=True)
            report(name, w, pg, msgs, r.status if r else None)
        if w in ("d", "m"):                                                       # states that need an interaction
            msgs.clear()
            pg.goto(base + "/profile", wait_until="networkidle")
            pg.fill("#years", "abc")
            pg.click("form[action='/profile'] button[type=submit]")
            pg.wait_for_load_state("networkidle")
            pg.screenshot(path=str(out / f"profile_error_{w}.png"), full_page=True)
            report("profile_error", w, pg, msgs)
            msgs.clear()
            pg.goto(base + "/account", wait_until="networkidle")
            pg.fill("#current", "not the password")
            pg.fill("#new", "another password 77")
            pg.fill("#confirm", "another password 77")
            pg.click("form[action='/account/password'] button[type=submit]")
            pg.wait_for_load_state("networkidle")
            pg.screenshot(path=str(out / f"account_error_{w}.png"), full_page=True)
            report("account_error", w, pg, msgs)
        ctx.close()
    browser.close()
sys.exit(1 if problems else 0)
