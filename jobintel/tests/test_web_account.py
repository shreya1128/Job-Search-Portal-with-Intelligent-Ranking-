"""Accounts and the pieces that need a logged-in user: CSRF on every POST, registration and login rules, access control, saving jobs, and the fit panel on a job page."""
import re

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from jobintel.db import models as M

from .web_fixtures import BASE_ID, csrf_of, make_app, register


@pytest.fixture()
def site(engine, tmp_path):
    app, be = make_app(engine, tmp_path)
    return app, be, app.test_client(), engine


def login_as(c, username="tester", password="correct horse 42"):
    return c.post("/login", data={"csrf_token": csrf_of(c), "identifier": username, "password": password})


# ------------------------------------------------------------------------------------------------------------------------ CSRF
@pytest.mark.parametrize("path,data", [("/login", {"identifier": "a", "password": "b"}), ("/register", {"username": "abc", "email": "a@b.co", "password": "x" * 9, "confirm": "x" * 9}),
                                       ("/logout", {}), ("/dashboard/profile", {"skills": "sql"})])
def test_post_without_csrf_token_is_refused(site, path, data):
    _, _, c, _ = site
    r = c.post(path, data=data)
    assert r.status_code == 400 and "expired or is not valid" in r.data.decode()


def test_post_with_wrong_csrf_token_is_refused(site):
    _, _, c, _ = site
    csrf_of(c)
    assert c.post("/login", data={"csrf_token": "forged", "identifier": "a", "password": "b"}).status_code == 400


def test_api_interact_requires_login_then_csrf(site):
    _, _, c, _ = site
    assert c.post("/api/interact", json={"job_id": BASE_ID, "action": "save"}).status_code in (400, 401)
    register(c)
    assert c.post("/api/interact", json={"job_id": BASE_ID, "action": "save"}).status_code == 400                         # logged in but no token
    tok = csrf_of(c)
    ok = c.post("/api/interact", json={"job_id": BASE_ID, "action": "save"}, headers={"X-CSRF-Token": tok})
    assert ok.status_code == 200 and ok.get_json()["ok"] is True


# ------------------------------------------------------------------------------------------------------------------------ registration and login
def test_register_creates_account_and_logs_in(site):
    _, _, c, engine = site
    r = register(c)
    assert r.status_code == 302 and r.headers["Location"].endswith("/dashboard")
    with Session(engine) as s:
        u = s.execute(select(M.User)).scalar_one()
        assert u.username == "tester" and u.password_hash != "correct horse 42" and u.password_hash and not u.is_synthetic
    assert "My workspace" in c.get("/dashboard").data.decode()


@pytest.mark.parametrize("kw,message", [(dict(username="ab"), "User name"), (dict(email="not-an-email"), "valid e-mail"), (dict(password="short", ), "at least 8"),
                                         (dict(username="synthetic_bot"), "reserved")])
def test_registration_validation(site, kw, message):
    _, _, c, engine = site
    args = dict(username="tester", email="t@example.com", password="correct horse 42")
    args.update(kw)
    r = register(c, **args)
    assert r.status_code == 400 and message in r.data.decode()
    with Session(engine) as s:
        assert s.execute(select(func.count()).select_from(M.User)).scalar_one() == 0


def test_duplicate_registration_is_refused(site):
    _, _, c, _ = site
    register(c)
    c.post("/logout", data={"csrf_token": csrf_of(c)})
    r = register(c)
    assert r.status_code == 400 and "already registered" in r.data.decode()


def test_login_wrong_password_and_throttle(site):
    app, _, c, _ = site
    register(c)
    c.post("/logout", data={"csrf_token": csrf_of(c)})
    bad = login_as(c, password="wrong password")
    assert bad.status_code == 401 and "not correct" in bad.data.decode() and "wrong password" not in bad.data.decode()
    codes = [login_as(c, password="wrong password").status_code for _ in range(10)]
    assert 429 in codes                                                                                                  # too many failures are blocked
    assert login_as(c).status_code == 429                                                                                # even the right password, until the window passes


def test_login_and_logout_roundtrip(site):
    _, _, c, _ = site
    register(c)
    assert c.post("/logout", data={"csrf_token": csrf_of(c)}).status_code == 302
    assert c.get("/dashboard").status_code == 302
    assert login_as(c).status_code == 302 and c.get("/dashboard").status_code == 200


def test_login_redirect_only_stays_on_this_site(site):
    _, _, c, _ = site
    register(c)
    c.post("/logout", data={"csrf_token": csrf_of(c)})
    r = c.post("/login", data={"csrf_token": csrf_of(c), "identifier": "tester", "password": "correct horse 42", "next": "https://evil.example/steal"})
    assert r.status_code == 302 and "evil.example" not in r.headers["Location"]


def test_dashboard_needs_login_and_returns_to_it(site):
    _, _, c, _ = site
    r = c.get("/dashboard")
    assert r.status_code == 302 and "/login?next=" in r.headers["Location"]


def test_synthetic_users_cannot_log_in(site):
    _, _, c, engine = site
    from werkzeug.security import generate_password_hash
    with Session(engine) as s:
        s.add(M.User(username="synthetic_1", email="s@example.com", password_hash=generate_password_hash("pw-pw-pw-pw"), is_synthetic=True))
        s.commit()
    assert login_as(c, "synthetic_1", "pw-pw-pw-pw").status_code == 401


# ------------------------------------------------------------------------------------------------------------------------ saving jobs and the fit panel
def test_saved_state_shows_on_search_and_job_pages(site):
    _, _, c, _ = site
    register(c)
    tok = csrf_of(c)
    assert c.post("/api/interact", json={"job_id": BASE_ID, "action": "save"}, headers={"X-CSRF-Token": tok}).status_code == 200
    assert 'btn-state-on' in c.get("/jobs?q=data+analyst").data.decode()
    assert re.search(r'btn-state-on[^>]*data-interact="save"', c.get(f"/job/{BASE_ID}").data.decode())
    c.post("/api/interact", json={"job_id": BASE_ID, "action": "unsave"}, headers={"X-CSRF-Token": tok})
    assert 'btn-state-on' not in c.get(f"/job/{BASE_ID}").data.decode()


def test_interact_validates_input(site):
    _, _, c, _ = site
    register(c)
    h = {"X-CSRF-Token": csrf_of(c)}
    assert c.post("/api/interact", json={"job_id": "1", "action": "save"}, headers=h).status_code == 400
    assert c.post("/api/interact", json={"job_id": BASE_ID, "action": "delete_everything"}, headers=h).status_code == 400
    assert c.post("/api/interact", json={"job_id": 99999999, "action": "save"}, headers=h).status_code == 404
    assert c.post("/api/interact", json={"job_id": True, "action": "save"}, headers=h).status_code == 400


def test_interactions_are_real_never_synthetic(site):
    _, _, c, engine = site
    register(c)
    c.post("/api/interact", json={"job_id": BASE_ID, "action": "save"}, headers={"X-CSRF-Token": csrf_of(c)})
    with Session(engine) as s:
        assert s.execute(select(M.JobInteraction.is_synthetic)).scalar_one() is False


def test_fit_panel_uses_the_saved_profile(site):
    _, _, c, _ = site
    register(c)
    r = c.post("/dashboard/profile", data={"csrf_token": csrf_of(c), "skills": "sql, excel", "years": "2", "role": "analyst", "city": "Pune"})
    assert r.status_code == 302
    html = c.get(f"/job/{BASE_ID}").data.decode()                                         # Data Analyst: sql, excel, power bi, data analysis; 1 to 3 years
    assert "You list <strong>2 of the 4</strong>" in html and "fit the 1 to 3 years" in html
    assert "Compared with your saved profile" in html


def test_viewing_a_job_is_recorded_only_for_logged_in_users(site):
    _, _, c, engine = site
    c.get(f"/job/{BASE_ID}")
    with Session(engine) as s:
        assert s.execute(select(func.count()).select_from(M.JobInteraction)).scalar_one() == 0
    register(c)
    c.get(f"/job/{BASE_ID}")
    with Session(engine) as s:
        assert s.execute(select(M.JobInteraction.viewed)).scalar_one() is True
