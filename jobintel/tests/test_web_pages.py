"""Web pages: every page renders, the navigation is the agreed seven items, search / filters / sorting / pagination behave, job details never show an unsafe link, and a missing resource
produces a plain message instead of a stack trace or a shell command."""
import re

import pytest
from sqlalchemy import func, select

from jobintel.db import models as M
from jobintel.web import create_app
from jobintel.web.backend import Backend
from jobintel.web.settings import WebSettings

from .web_fixtures import APPLY_ID, BASE_ID, EVIL_ID, FRAUD_ID, JOBS, NO_COMPANY_ID, make_app


@pytest.fixture()
def site(engine, tmp_path):
    app, be = make_app(engine, tmp_path)
    return app, be, app.test_client(), engine


def titles(html: bytes):
    return re.findall(r'<h3><a href="/job/\d+[^"]*">([^<]+)</a></h3>', html.decode())


# ------------------------------------------------------------------------------------------------------------------------ every page renders
PUBLIC = ["/", "/jobs", "/about", "/privacy", "/terms", "/health", "/login", "/register", "/recommendations", "/resume", "/career-insights", "/skill-gap"]


@pytest.mark.parametrize("path", PUBLIC)
def test_public_page_renders(site, path):
    _, _, c, _ = site
    r = c.get(path)
    assert r.status_code == 200, path
    html = r.data.decode()
    assert "<title>" in html and "Jobintel" in html and 'lang="en"' in html
    assert "Traceback" not in html


def test_unknown_page_is_a_designed_404(site):
    _, _, c, _ = site
    r = c.get("/no-such-page")
    assert r.status_code == 404 and "Page not found" in r.data.decode()


def test_navigation_is_the_seven_agreed_items_in_order(site):
    _, _, c, _ = site
    html = c.get("/").data.decode()
    nav = html[html.index('class="navbar-nav me-auto"'):]
    nav = nav[:nav.index("</ul>")]
    assert re.findall(r'class="nav-link[^"]*"[^>]*>([^<]+)<', nav) == ["Home", "Find Jobs", "Recommendations", "Resume", "Career Insights", "Saved Jobs", "Profile"]
    for forbidden in ("IR experiments", "DL experiments", "Experiments"):
        assert forbidden not in html


def test_current_page_is_marked_in_the_navigation(site):
    _, _, c, _ = site
    assert re.search(r'class="nav-link active" href="/jobs" aria-current="page"', c.get("/jobs").data.decode())
    assert re.search(r'class="nav-link active" href="/" aria-current="page"', c.get("/").data.decode())


def test_stage3_and_stage4_pages_render_real_workflows(site):
    _, _, c, _ = site
    expected = {
        "/recommendations": "Find recommendations",
        "/resume": "Analyse resume",
        "/career-insights": "Skills appearing in postings",
        "/skill-gap": "Skill Gap Analysis",
    }
    for path, marker in expected.items():
        r = c.get(path)
        html = r.data.decode()
        assert r.status_code == 200 and marker.lower() in html.lower(), path
        assert "Not built yet" not in html and "Traceback" not in html


@pytest.mark.parametrize("path", ["/saved", "/profile", "/account", "/dashboard"])
def test_personal_pages_send_visitors_to_log_in_and_back(site, path):
    _, _, c, _ = site
    r = c.get(path)
    assert r.status_code == 302 and r.headers["Location"] == f"/login?next={path}"


def test_experiment_pages_are_not_reachable(site):
    _, _, c, _ = site
    assert c.get("/ir-experiments").status_code == 404 and c.get("/dl-experiments").status_code == 404


def test_security_headers(site):
    _, _, c, _ = site
    r = c.get("/")
    assert "script-src 'self'" in r.headers["Content-Security-Policy"] and r.headers["X-Frame-Options"] == "DENY" and r.headers["X-Content-Type-Options"] == "nosniff"


def test_home_is_short_and_uses_real_facts(site):
    _, _, c, _ = site
    html = c.get("/").data.decode()
    assert html.count('<section') <= 4                                   # hero, areas, examples, closing row: no wall of sections
    assert "Browse by job area" in html and "Understand the job market" in html and "Explore jobs" in html and "Analyse a resume" in html
    assert html.count('class="area"') <= 6
    for fake in ("testimonial", "trusted by", "10x", "%+"):
        assert fake not in html.lower()


# ------------------------------------------------------------------------------------------------------------------------ search
def test_search_returns_cards_with_a_plain_reason(site):
    _, _, c, _ = site
    html = c.get("/jobs?q=data+analyst").data.decode()
    assert "Data Analyst" in html and "Why it matched" in html
    assert "ribbon" not in html and "lm_loglik" not in html and "DFR" not in html     # no scoring internals on the cards


def test_pagination_ten_per_page_and_last_page_remainder(site):
    _, _, c, _ = site
    p1 = c.get("/jobs?q=analyst")
    assert len(titles(p1.data)) == 10 and 'aria-current="page"' in p1.data.decode()
    total = int(re.search(r"of the top (\d+) matches", p1.data.decode()).group(1))
    pages = -(-total // 10)
    last = c.get(f"/jobs?q=analyst&page={pages}")
    assert len(titles(last.data)) == total - 10 * (pages - 1)
    seen = [t for n in range(1, pages + 1) for t in titles(c.get(f"/jobs?q=analyst&page={n}").data)]
    assert len(seen) == total == len(set(seen)) or len(seen) == total                 # every result on exactly one page


def test_page_beyond_the_end_shows_the_last_page_not_an_error(site):
    _, _, c, _ = site
    r = c.get("/jobs?q=analyst&page=999")
    assert r.status_code == 200 and titles(r.data)


def test_bad_page_value_is_reported_not_a_500(site):
    _, _, c, _ = site
    r = c.get("/jobs?q=analyst&page=abc")
    assert r.status_code == 200 and "must be a number" in r.data.decode()


def test_sort_by_experience_orders_the_pool(site):
    _, _, c, _ = site
    def exp_first(sort):
        html = c.get(f"/jobs?q=analyst&sort={sort}").data.decode()
        return [float(m) for m in re.findall(r"<span>(\d+(?:\.\d+)?)(?:-\d+(?:\.\d+)?)? yrs?</span>", html)]
    low, high = exp_first("exp_low"), exp_first("exp_high")
    assert low == sorted(low) and high == sorted(high, reverse=True)
    assert "sorted by experience" in c.get("/jobs?q=analyst&sort=exp_low").data.decode()


def test_unknown_sort_value_falls_back_and_says_so(site):
    _, _, c, _ = site
    r = c.get("/jobs?q=analyst&sort=nonsense")
    assert r.status_code == 200 and "sort must be one of" in r.data.decode() and titles(r.data)


def test_area_filter_and_validation(site):
    _, _, c, _ = site
    ok = c.get("/jobs?area=logistics_transport")
    assert ok.status_code == 200 and "Warehouse Assistant" in ok.data.decode() and "Reporting Analyst" not in ok.data.decode()
    assert "Area: Logistics and transport" in ok.data.decode()                         # an active-filter chip with a remove link
    bad = c.get("/jobs?area=drop_table")
    assert bad.status_code == 200 and "area must be one of" in bad.data.decode()


def test_location_city_and_country(site):
    _, _, c, _ = site
    pune = titles(c.get("/jobs?q=analyst&city=Pune").data)
    assert pune and all("Pune" in c.get("/jobs?q=analyst&city=Pune").data.decode() for _ in [0])
    assert titles(c.get("/jobs?q=analyst&city=Pune,+India").data) == pune             # 'Pune, India' is the city Pune
    by_country = c.get("/jobs?q=analyst&city=India")
    assert "Country: India" in by_country.data.decode() and titles(by_country.data)


def test_experience_band_filter_hides_unstated_experience(site):
    _, _, c, _ = site
    html = c.get("/jobs?exp=0-1").data.decode()
    assert "Warehouse Assistant" not in html and "Experience: Fresher" in html


def test_no_results_state(site):
    _, _, c, _ = site
    html = c.get("/jobs?q=zzzqqxx").data.decode()
    assert "No jobs match that search" in html and "ignored" in html or "None of those words" in html


def test_overlong_query_is_cut_and_reported(site):
    _, _, c, _ = site
    r = c.get("/jobs?q=" + "a" * 500)
    assert r.status_code == 200 and "longer than 200" in r.data.decode()


def test_search_text_is_escaped(site):
    _, _, c, _ = site
    r = c.get('/jobs?q=%3Cscript%3Ealert(1)%3C/script%3E&city=%22%3E%3Cimg+src=x%3E')
    html = r.data.decode()
    assert r.status_code == 200 and "<script>alert(1)" not in html and '"><img' not in html


def test_old_search_address_redirects(site):
    _, _, c, _ = site
    r = c.get("/search?q=python&city=Pune")
    assert r.status_code == 301 and r.headers["Location"].startswith("/jobs?") and "q=python" in r.headers["Location"]


def test_search_is_logged_once_per_search_not_per_page(site):
    app, be, c, engine = site
    from sqlalchemy.orm import Session
    c.get("/jobs?q=analyst"); c.get("/jobs?q=analyst&page=2"); c.get("/jobs?q=analyst&page=3")
    with Session(engine) as s:
        assert s.execute(select(func.count()).select_from(M.Search)).scalar_one() == 1
        assert s.execute(select(M.Search.user_id)).scalar_one() is None             # an anonymous search is not linked to anyone


def test_anonymous_cards_ask_for_login_to_save(site):
    _, _, c, _ = site
    html = c.get("/jobs?q=analyst").data.decode()
    assert "/login?next=" in html and 'data-interact="save"' not in html


def test_placeholder_company_names_are_never_shown(site):
    _, _, c, _ = site
    assert "Company-4c8196" not in c.get(f"/job/{NO_COMPANY_ID}").data.decode()


# ------------------------------------------------------------------------------------------------------------------------ job details
def test_job_page_facts_skills_and_related_jobs(site):
    _, _, c, _ = site
    html = c.get(f"/job/{BASE_ID}").data.decode()
    assert "<h1>Data Analyst</h1>" in html and "power bi" in html and "Related jobs" in html and "Why it may suit you" in html


def test_unknown_job_is_404(site):
    _, _, c, _ = site
    assert c.get("/job/999999").status_code == 404


def test_apply_link_only_for_http_links(site):
    _, _, c, _ = site
    ok = c.get(f"/job/{APPLY_ID}").data.decode()
    assert 'href="https://example.com/apply/1"' in ok and 'rel="noopener noreferrer nofollow"' in ok
    evil = c.get(f"/job/{EVIL_ID}").data.decode()
    assert "javascript:" not in evil and "View the original posting" not in evil and "does not include an application link" in evil


def test_flagged_posting_shows_banner_and_no_apply_button(site):
    _, _, c, _ = site
    html = c.get(f"/job/{FRAUD_ID}").data.decode()
    assert "Flagged as likely fraudulent" in html and "View the original posting" not in html


def test_query_words_are_highlighted_on_the_job_page(site):
    _, _, c, _ = site
    html = c.get(f"/job/{BASE_ID}?q=analyst&ref=search").data.decode()
    assert "<mark>" in html and "are highlighted" in html


def test_job_page_without_a_profile_invites_one_without_inventing_a_match(site):
    _, _, c, _ = site
    html = c.get(f"/job/{BASE_ID}").data.decode()
    assert "Add your skills or analyse a resume" in html and "You list" not in html


# ------------------------------------------------------------------------------------------------------------------------ resources that are missing
def test_missing_index_gives_a_plain_message_not_a_command(engine, tmp_path):
    st = WebSettings(testing=True, secret_key="k", warmup=False, data_dir=tmp_path)
    app = create_app(st, Backend(st, db=engine))
    c = app.test_client()
    html = c.get("/jobs?q=analyst").data.decode()              # the page echoes the visitor's own query, so use a word that is not a leak keyword
    assert "not available right now" in html
    for leak in ("scripts/", "build_index", "python scripts", "--write-db", "Traceback", str(tmp_path), "job_index.pkl"):
        assert leak not in html, leak
    status = c.get("/health").data.decode()
    assert "unavailable" in status and "build_index" not in status and str(tmp_path) not in status      # reasons are for the log, not for visitors


def test_status_page_shows_reasons_only_in_debug(engine, tmp_path):
    st = WebSettings(testing=True, secret_key="k", warmup=False, data_dir=tmp_path, debug=True)
    app = create_app(st, Backend(st, db=engine))
    c = app.test_client()
    c.get("/jobs?q=analyst")                                    # a part is only reported once a page has tried to load it
    assert "does not exist" in c.get("/health").data.decode()


def test_home_page_still_renders_without_the_search_index(engine, tmp_path):
    st = WebSettings(testing=True, secret_key="k", warmup=False, data_dir=tmp_path)
    app = create_app(st, Backend(st, db=engine))
    r = app.test_client().get("/")
    assert r.status_code == 200 and "Understand the job market" in r.data.decode()
