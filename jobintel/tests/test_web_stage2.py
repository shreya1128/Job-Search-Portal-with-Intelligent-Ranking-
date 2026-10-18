"""Stage 2 of the web application: Saved Jobs, the Profile page, the personal workspace and account settings (change password, download data, delete account).
The things that must hold: nobody sees or changes another user's data, every POST needs the CSRF token, input is validated and echoed back escaped, profile saving never touches the
resume fields or the source of a stored skill, a changed password logs the other browsers out, and deleting an account removes everything stored about it and nothing else."""
import json
import re

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from jobintel.db import models as M
from jobintel.web import create_app
from jobintel.web.backend import Backend
from jobintel.web.settings import WebSettings

from .web_fixtures import BASE_ID, FRAUD_ID, NO_COMPANY_ID, csrf_of, make_app, register

PW = "correct horse 42"


@pytest.fixture()
def site(engine, tmp_path):
    app, be = make_app(engine, tmp_path)
    return app, be, app.test_client(), engine


def post(c, path, **data):
    return c.post(path, data={"csrf_token": csrf_of(c), **data})


def interact(c, job_id, action):
    r = c.post("/api/interact", json={"job_id": job_id, "action": action}, headers={"X-CSRF-Token": csrf_of(c)})
    assert r.status_code == 200, r.data
    return r


def second_user(app, name="other"):
    c2 = app.test_client()
    register(c2, name, f"{name}@example.com", PW)
    return c2


def ids_on(html: bytes):
    return [int(x) for x in re.findall(r'<h3><a href="/job/(\d+)', html.decode())]


def rows(engine, model, *where):
    with Session(engine) as s:
        return s.execute(select(model).where(*where)).scalars().all()


def count(engine, model, *where):
    with Session(engine) as s:
        return int(s.execute(select(func.count()).select_from(model).where(*where)).scalar_one())


def uid(engine, username):
    with Session(engine) as s:
        return s.execute(select(M.User.user_id).where(M.User.username == username)).scalar_one()


def flashed(c):
    """The flash messages waiting in the session (read without consuming them)."""
    with c.session_transaction() as sess:
        return [m for _, m in sess.get("_flashes", [])]


# ================================================================================================================================ Saved Jobs
def test_saved_page_for_a_new_user_says_nothing_is_saved(site):
    _, _, c, _ = site
    register(c)
    html = c.get("/saved").data.decode()
    assert "Nothing saved yet" in html and 'href="/jobs"' in html


def test_saved_page_lists_only_saved_jobs_with_real_details(site):
    _, _, c, _ = site
    register(c)
    interact(c, BASE_ID, "save")
    interact(c, BASE_ID + 3, "save")
    c.get(f"/job/{BASE_ID + 4}")                                           # viewed, not saved
    html = c.get("/saved").data
    assert sorted(ids_on(html)) == [BASE_ID, BASE_ID + 3]
    text = html.decode()
    assert "Data Analyst" in text and "Machine Learning Engineer" in text and "Backend Developer" not in text
    assert "Pune" in text and "1 to 3 years" in text and "Last activity" in text
    assert "power bi" in text                                               # skills come from the posting's own skill rows
    assert "status-chip" not in text                                        # no status is shown that was never set


def test_saved_page_hides_placeholder_company_names_and_marks_flagged_postings(site):
    _, _, c, _ = site
    register(c)
    interact(c, NO_COMPANY_ID, "save")
    interact(c, FRAUD_ID, "save")
    text = c.get("/saved").data.decode()
    assert "Company-4c8196" not in text and "likely fraudulent" in text


def test_saved_skills_you_list_are_highlighted(site):
    _, _, c, _ = site
    register(c)
    interact(c, BASE_ID, "save")
    assert "chip-have" not in c.get("/saved").data.decode()
    post(c, "/profile", skills="sql, excel", years="2")
    html = c.get("/saved").data.decode()
    assert re.search(r'chip-have"><span class="visually-hidden">you list: </span>sql', html) and "Skills you list on your" not in html


def test_mark_applied_and_remove_the_mark(site):
    _, _, c, engine = site
    register(c)
    interact(c, BASE_ID, "save")
    r = post(c, f"/saved/{BASE_ID}/applied", value="1", next="/saved")
    assert r.status_code == 302 and r.headers["Location"] == "/saved"
    html = c.get("/saved").data.decode()
    assert 'status-chip">Marked as applied' in html and "Remove applied mark" in html
    post(c, f"/saved/{BASE_ID}/applied", value="0")
    html = c.get("/saved").data.decode()
    assert "Remove applied mark" not in html and "status-chip" not in html and "Mark as applied" in html
    assert rows(engine, M.JobInteraction)[0].applied is False


def test_remove_from_saved_keeps_the_viewing_history_and_empties_the_page(site):
    _, _, c, engine = site
    register(c)
    interact(c, BASE_ID, "save")
    r = post(c, f"/saved/{BASE_ID}/remove", next="/saved")
    assert r.status_code == 302
    assert any("Removed" in m for m in flashed(c))                         # read before the next page consumes the message
    row = rows(engine, M.JobInteraction)[0]
    assert row.saved is False and row.viewed is True                       # unsaved, but it was still viewed
    assert "Nothing saved yet" in c.get("/saved").data.decode()


def test_removing_or_marking_a_job_that_is_not_saved_changes_nothing(site):
    _, _, c, engine = site
    register(c)
    for path, extra in ((f"/saved/{BASE_ID}/remove", {}), (f"/saved/{BASE_ID}/applied", {"value": "1"}), ("/saved/987654/remove", {})):
        r = post(c, path, **extra)
        assert r.status_code == 302
        assert any("not in your saved list" in m for m in flashed(c))
    assert count(engine, M.JobInteraction) == 0                            # no stray rows were created


def test_saved_actions_need_the_csrf_token(site):
    _, _, c, engine = site
    register(c)
    interact(c, BASE_ID, "save")
    assert c.post(f"/saved/{BASE_ID}/remove").status_code == 400
    assert c.post(f"/saved/{BASE_ID}/applied", data={"value": "1"}).status_code == 400
    assert c.post(f"/saved/{BASE_ID}/remove", data={"csrf_token": "forged"}).status_code == 400
    assert rows(engine, M.JobInteraction)[0].saved is True


def test_saved_actions_need_a_login(site):
    _, _, c, _ = site
    r = c.post(f"/saved/{BASE_ID}/remove", data={"csrf_token": csrf_of(c)})
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_after_an_action_the_user_returns_only_to_a_page_on_this_site(site):
    _, _, c, _ = site
    register(c)
    interact(c, BASE_ID, "save")
    assert post(c, f"/saved/{BASE_ID}/applied", value="1", next="/saved?status=applied").headers["Location"] == "/saved?status=applied"
    for evil in ("https://evil.example/x", "//evil.example/x", "/\\evil.example", ""):
        r = post(c, f"/saved/{BASE_ID}/applied", value="1", next=evil)
        assert r.status_code == 302 and r.headers["Location"] == "/saved", evil


def test_users_never_see_or_change_each_others_saved_jobs(site):
    app, _, c, engine = site
    register(c, "alice", "alice@example.com", PW)
    interact(c, BASE_ID, "save")
    bob = second_user(app, "bob")
    assert "Nothing saved yet" in bob.get("/saved").data.decode()
    interact(bob, BASE_ID + 1, "save")
    assert ids_on(bob.get("/saved").data) == [BASE_ID + 1] and ids_on(c.get("/saved").data) == [BASE_ID]
    r = post(bob, f"/saved/{BASE_ID}/remove")                                # Bob tries to remove Alice's saved job
    assert r.status_code == 302 and any("not in your saved list" in m for m in flashed(bob))
    post(bob, f"/saved/{BASE_ID}/applied", value="1")
    a = rows(engine, M.JobInteraction, M.JobInteraction.user_id == uid(engine, "alice"))[0]
    assert a.saved is True and a.applied is False


def test_status_filter_and_bad_status(site):
    _, _, c, _ = site
    register(c)
    for j in (BASE_ID, BASE_ID + 1, BASE_ID + 3):
        interact(c, j, "save")
    post(c, f"/saved/{BASE_ID + 1}/applied", value="1")
    assert ids_on(c.get("/saved?status=applied").data) == [BASE_ID + 1]
    assert sorted(ids_on(c.get("/saved?status=open").data)) == [BASE_ID, BASE_ID + 3]
    assert len(ids_on(c.get("/saved?status=all").data)) == 3
    bad = c.get("/saved?status=%27%3Bdrop")
    assert bad.status_code == 200 and "must be one of" in bad.data.decode() and len(ids_on(bad.data)) == 3
    none = c.get("/saved?status=applied").data.decode()
    post(c, f"/saved/{BASE_ID + 1}/applied", value="0")
    assert "No saved job matches this view" in c.get("/saved?status=applied").data.decode() and none


def test_saved_pagination(site):
    _, _, c, _ = site
    register(c)
    for j in range(BASE_ID, BASE_ID + 12):
        interact(c, j, "save")
    p1, p2 = c.get("/saved"), c.get("/saved?page=2")
    assert len(ids_on(p1.data)) == 10 and len(ids_on(p2.data)) == 2
    assert set(ids_on(p1.data)).isdisjoint(ids_on(p2.data))
    assert "Showing 1&ndash;10" in p1.data.decode() and 'rel="next"' in p1.data.decode()
    assert ids_on(c.get("/saved?page=999").data) == ids_on(p2.data)           # beyond the end shows the last page
    assert c.get("/saved?page=abc").status_code == 200


# ================================================================================================================================ Profile
def stored_profile(engine, username="tester"):
    with Session(engine) as s:
        u = uid(engine, username)
        up = s.execute(select(M.UserProfile).where(M.UserProfile.user_id == u)).scalar_one_or_none()
        sk = {n: src for n, src in s.execute(select(M.Skill.canonical_name, M.UserSkill.source).join(M.UserSkill, M.UserSkill.skill_id == M.Skill.skill_id).where(M.UserSkill.user_id == u))}
        return up, sk


def test_profile_page_for_a_new_user_is_empty_and_honest(site):
    _, _, c, _ = site
    register(c)
    html = c.get("/profile").data.decode()
    assert "0 of 6 done" in html and "Not stored yet" in html and "Certifications" in html
    assert 'id="role"' in html and 'value=""' in html


def test_profile_saves_and_reloads_every_field(site):
    _, _, c, engine = site
    register(c)
    r = post(c, "/profile", full_name="Asha Rao", education="B.Sc. Statistics", summary="Analyst with two years of reporting.", role="data analyst", location="Pune",
             years="2", skills="SQL, excel")
    assert r.status_code == 302 and r.headers["Location"] == "/profile"
    assert any("Profile saved with 2 skills" in m for m in flashed(c))
    html = c.get("/profile").data.decode()
    for needle in ('value="Asha Rao"', 'value="B.Sc. Statistics"', 'value="data analyst"', 'value="Pune"', 'value="2"', "Analyst with two years of reporting.", "excel, sql"):
        assert needle in html, needle
    assert "5 of 6 done" in html                                                # skills, years, role, location, education; the resume step is not done
    up, sk = stored_profile(engine)
    assert (up.full_name, up.education, up.preferred_role, up.preferred_location, up.experience_years) == ("Asha Rao", "B.Sc. Statistics", "data analyst", "Pune", 2.0)
    assert sk == {"sql": "manual", "excel": "manual"}


def test_zero_years_is_a_real_answer_and_blank_is_unknown(site):
    _, _, c, engine = site
    register(c)
    post(c, "/profile", years="0")
    assert stored_profile(engine)[0].experience_years == 0.0 and "Years of experience" in c.get("/profile").data.decode()
    assert "years" in c.get("/profile").data.decode() and 'value="0"' in c.get("/profile").data.decode()
    post(c, "/profile", years="")
    assert stored_profile(engine)[0].experience_years is None


@pytest.mark.parametrize("field,value,message", [("years", "-1", "between 0 and 60"), ("years", "61", "between 0 and 60"), ("years", "abc", "must be a number"),
                                                  ("years", "nan", "between 0 and 60"), ("full_name", "x" * 121, "longer than 120"), ("summary", "y" * 1001, "longer than 1000"),
                                                  ("role", "z" * 121, "longer than 120")])
def test_profile_validation_refuses_and_keeps_what_was_typed(site, field, value, message):
    _, _, c, engine = site
    register(c)
    post(c, "/profile", role="old role", skills="sql")
    r = post(c, "/profile", **{"location": "Goa", field: value})
    html = r.data.decode()
    assert r.status_code == 400 and message in html and "Please fix" in html
    assert stored_profile(engine)[0].preferred_role == "old role" and stored_profile(engine)[0].preferred_location is None      # nothing was saved
    assert 'value="Goa"' in html                                               # what was typed is still in the form


def test_unknown_skills_are_reported_and_not_stored(site):
    _, _, c, engine = site
    register(c)
    post(c, "/profile", skills="sql, underwater basket weaving, javascript")
    msg = " ".join(flashed(c))
    assert "Profile saved with 1 skill." in msg and "underwater basket weaving" in msg and "javascript" in msg and "not stored" in msg
    assert list(stored_profile(engine)[1]) == ["sql"]


def test_removing_a_skill_deletes_it_and_a_resume_skill_keeps_its_source(site):
    _, _, c, engine = site
    register(c)
    post(c, "/profile", skills="sql, excel, python")
    with Session(engine) as s:                                                  # a skill that the Resume page (later) read from a resume
        s.execute(M.UserSkill.__table__.update().where(M.UserSkill.user_id == uid(engine, "tester"), M.UserSkill.skill_id == s.execute(
            select(M.Skill.skill_id).where(M.Skill.canonical_name == "sql")).scalar_one()).values(source="resume"))
        s.commit()
    assert "Skills read from a resume" in c.get("/profile").data.decode()
    post(c, "/profile", skills="sql, python")                                   # excel removed, sql kept
    assert stored_profile(engine)[1] == {"sql": "resume", "python": "manual"}


def test_saving_the_profile_never_touches_the_resume_fields(site):
    _, _, c, engine = site
    register(c)
    with Session(engine) as s:
        s.add(M.UserProfile(user_id=uid(engine, "tester"), resume_filename="cv.pdf", resume_text="Fictional resume text.", preferred_categories=["IT"]))
        s.commit()
    post(c, "/profile", role="analyst", skills="sql")
    up, _ = stored_profile(engine)
    assert (up.resume_filename, up.resume_text, up.preferred_categories, up.preferred_role) == ("cv.pdf", "Fictional resume text.", ["IT"], "analyst")
    assert "6 of 6 done" not in c.get("/profile").data.decode() and "Resume analysed" in c.get("/profile").data.decode()


def test_profile_text_is_escaped(site):
    _, _, c, _ = site
    register(c)
    post(c, "/profile", full_name="<script>alert(1)</script>", summary="</textarea><img src=x onerror=alert(1)>", role='"><b>x</b>')
    html = c.get("/profile").data.decode()
    assert "<script>alert(1)" not in html and "<img src=x" not in html and '"><b>x</b>' not in html and "&lt;script&gt;" in html


def test_profile_needs_login_and_csrf(site):
    _, _, c, _ = site
    r = c.post("/profile", data={"csrf_token": csrf_of(c), "role": "x"})
    assert r.status_code == 302 and "/login" in r.headers["Location"]
    register(c)
    assert c.post("/profile", data={"role": "x"}).status_code == 400


def test_profiles_are_separate_per_user(site):
    app, _, c, engine = site
    register(c, "alice", "alice@example.com", PW)
    post(c, "/profile", role="alice role", skills="sql")
    bob = second_user(app, "bob")
    html = bob.get("/profile").data.decode()
    assert "alice role" not in html and "0 of 6 done" in html
    post(bob, "/profile", role="bob role", skills="python")
    assert stored_profile(engine, "alice")[0].preferred_role == "alice role" and list(stored_profile(engine, "alice")[1]) == ["sql"]


def test_the_old_profile_form_address_still_works(site):
    _, _, c, engine = site
    register(c)
    r = post(c, "/dashboard/profile", role="analyst", skills="sql", years="1")
    assert r.status_code == 302 and stored_profile(engine)[0].preferred_role == "analyst"


def test_profile_saves_the_rest_when_the_skill_list_is_not_loaded(engine, tmp_path):
    st = WebSettings(testing=True, secret_key="k", warmup=False, data_dir=tmp_path)           # no search index: the skill list cannot be checked
    c = create_app(st, Backend(st, db=engine)).test_client()
    register(c)
    r = post(c, "/profile", role="analyst", skills="sql")
    assert r.status_code == 302 and stored_profile(engine)[0].preferred_role == "analyst"
    msg = " ".join(flashed(c))
    assert "skills were not changed" in msg and str(tmp_path) not in msg and "build_index" not in msg


# ================================================================================================================================ workspace
def test_workspace_for_a_new_user_shows_nothing_invented(site):
    _, _, c, _ = site
    register(c)
    html = c.get("/dashboard").data.decode()
    assert "0 of 6 steps done" in html and "Nothing saved." in html and "None marked." in html and "No searches yet." in html and "No resume analysed yet." in html
    assert "later stages" in html and "Recommended jobs" not in html
    for label in ("Jobs viewed", "Jobs saved", "Marked as applied"):
        assert re.search(rf'<div class="v">0</div><div class="l">{label}</div>', html), label


def test_workspace_reflects_real_activity(site):
    _, _, c, _ = site
    register(c)
    post(c, "/profile", role="analyst", years="1", skills="sql")
    interact(c, BASE_ID, "save")
    interact(c, BASE_ID + 1, "save")
    post(c, f"/saved/{BASE_ID}/applied", value="1")
    c.get(f"/job/{BASE_ID + 3}")
    c.get("/jobs?q=analyst")
    html = c.get("/dashboard").data.decode()
    assert "3 of 6 steps done" in html
    assert re.search(r'<div class="v">2</div><div class="l">Jobs saved</div>', html) and re.search(r'<div class="v">1</div><div class="l">Marked as applied</div>', html)
    assert "Applications you marked" in html and "Machine Learning Engineer" in html and 'href="/jobs?q=analyst"' in html and re.search(r"\d+ results, ", html)
    assert "Manage saved jobs" in html or "See all" in html


def test_workspace_shows_stored_recommendations_only_when_they_exist(site):
    _, _, c, engine = site
    register(c)
    assert "Recommended jobs" not in c.get("/dashboard").data.decode()
    with Session(engine) as s:
        s.add(M.Recommendation(user_id=uid(engine, "tester"), job_id=BASE_ID, position=1, final_score=0.5))
        s.commit()
    html = c.get("/dashboard").data.decode()
    assert "Recommended jobs" in html and "Data Analyst" in html and "from your last saved recommendation run" in html.lower()


# ================================================================================================================================ account settings
def test_account_page_needs_login_and_shows_no_secret(site):
    _, _, c, _ = site
    assert c.get("/account").status_code == 302
    register(c)
    html = c.get("/account").data.decode()
    assert "tester@example.com" in html and "Change password" in html and "Delete my account" in html and "scrypt" not in html and "pbkdf2" not in html


@pytest.mark.parametrize("path", ["/account/password", "/account/delete"])
def test_account_posts_need_the_csrf_token(site, path):
    _, _, c, _ = site
    register(c)
    assert c.post(path, data={"current": "x", "new": "y" * 9, "confirm": "y" * 9, "password": PW, "understood": "1"}).status_code == 400


def test_change_password_logs_in_with_the_new_one_and_logs_other_browsers_out(site):
    app, _, c, _ = site
    register(c)
    other = app.test_client()                                                     # a second browser, logged in with the old password
    other.post("/login", data={"csrf_token": csrf_of(other), "identifier": "tester", "password": PW})
    assert other.get("/dashboard").status_code == 200
    r = post(c, "/account/password", current=PW, new="a brand new pass 9", confirm="a brand new pass 9")
    assert r.status_code == 302 and any("Password changed" in m for m in flashed(c))
    assert c.get("/dashboard").status_code == 200                                 # this browser stays logged in
    r = other.get("/dashboard")
    assert r.status_code == 302 and "/login" in r.headers["Location"]             # the other one is logged out
    fresh = app.test_client()
    assert fresh.post("/login", data={"csrf_token": csrf_of(fresh), "identifier": "tester", "password": PW}).status_code == 401
    assert fresh.post("/login", data={"csrf_token": csrf_of(fresh), "identifier": "tester", "password": "a brand new pass 9"}).status_code == 302


@pytest.mark.parametrize("current,new,confirm,message", [("wrong password", "a brand new pass 9", "a brand new pass 9", "current password is not correct"),
                                                          (PW, "short", "short", "at least 8"), (PW, "a brand new pass 9", "different pass 99", "two new passwords differ"),
                                                          (PW, PW, PW, "different from the current"), (PW, "tester", "tester", "at least 8"),
                                                          (PW, "x" * 129, "x" * 129, "at most 128")])
def test_change_password_refusals_leave_the_password_alone(site, current, new, confirm, message):
    app, _, c, _ = site
    register(c)
    r = post(c, "/account/password", current=current, new=new, confirm=confirm)
    assert r.status_code == 400 and message in r.data.decode()
    fresh = app.test_client()
    assert fresh.post("/login", data={"csrf_token": csrf_of(fresh), "identifier": "tester", "password": PW}).status_code == 302


def test_change_password_cannot_be_used_to_guess_the_current_one(site):
    _, _, c, _ = site
    register(c)
    codes = [post(c, "/account/password", current=f"guess {n}", new="a brand new pass 9", confirm="a brand new pass 9").status_code for n in range(12)]
    assert 429 in codes
    assert post(c, "/account/password", current=PW, new="a brand new pass 9", confirm="a brand new pass 9").status_code == 429


def test_a_session_without_the_password_stamp_is_not_logged_in(site):
    _, _, c, engine = site
    register(c)
    with c.session_transaction() as sess:
        sess.pop("pwv")                                                           # e.g. a cookie issued before the stamp existed
    r = c.get("/dashboard")
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_download_my_data_contains_only_my_data(site):
    app, _, c, engine = site
    register(c, "alice", "alice@example.com", PW)
    post(c, "/profile", full_name="Alice A", role="analyst", location="Pune", years="2", skills="sql")
    interact(c, BASE_ID, "save")
    c.get("/jobs?q=analyst")
    bob = second_user(app, "bob")
    interact(bob, BASE_ID + 1, "save")
    post(bob, "/profile", full_name="Bob B", skills="python")
    r = c.get("/account/export")
    assert r.status_code == 200 and r.mimetype == "application/json" and "attachment" in r.headers["Content-Disposition"] and r.headers["Cache-Control"] == "no-store"
    data = json.loads(r.data)
    assert data["account"]["username"] == "alice" and data["profile"]["full_name"] == "Alice A" and data["profile"]["target_role"] == "analyst"
    assert data["skills"] == [{"skill": "sql", "source": "manual"}]
    assert [j["job_id"] for j in data["jobs"]] == [BASE_ID] and data["jobs"][0]["saved"] is True
    assert [x["query"] for x in data["searches"]] == ["analyst"]
    text = r.data.decode()
    for secret in ("bob", "Bob B", "password", "scrypt", "pbkdf2"):
        assert secret not in text, secret
    assert c.get("/account/export").status_code == 200 and app.test_client().get("/account/export").status_code == 302


def test_delete_needs_the_password_and_the_tick(site):
    _, _, c, engine = site
    register(c)
    r = post(c, "/account/delete", password=PW)
    assert r.status_code == 400 and "Tick the box" in r.data.decode()
    r = post(c, "/account/delete", password="wrong", understood="1")
    assert r.status_code == 400 and "password is not correct" in r.data.decode()
    assert count(engine, M.User) == 1 and c.get("/dashboard").status_code == 200


def test_delete_account_removes_everything_about_the_user_and_nothing_else(site):
    app, _, c, engine = site
    register(c, "alice", "alice@example.com", PW)
    post(c, "/profile", full_name="Alice", role="analyst", skills="sql, excel")
    interact(c, BASE_ID, "save")
    c.get("/jobs?q=analyst")
    a = uid(engine, "alice")
    with Session(engine) as s:
        s.add(M.Recommendation(user_id=a, job_id=BASE_ID, position=1, final_score=0.4))
        s.add(M.SkillGap(user_id=a, job_id=BASE_ID, skill_id=s.execute(select(M.Skill.skill_id).limit(1)).scalar_one(), status="missing"))
        s.commit()
    bob = second_user(app, "bob")
    post(bob, "/profile", role="bob role", skills="python")
    interact(bob, BASE_ID + 1, "save")
    bob.get("/jobs?q=data+analyst")
    b = uid(engine, "bob")
    results_before, jobs_before, skills_before = count(engine, M.SearchResult), count(engine, M.Job), count(engine, M.Skill)
    assert count(engine, M.SearchResult, M.SearchResult.search_id.in_(select(M.Search.search_id).where(M.Search.user_id == a))) > 0

    r = post(c, "/account/delete", password=PW, understood="1")
    assert r.status_code == 302 and r.headers["Location"] == "/"
    assert any("deleted" in m for m in flashed(c))
    for model in (M.UserProfile, M.UserSkill, M.JobInteraction, M.Search, M.Recommendation, M.SkillGap):
        assert count(engine, model, model.user_id == a) == 0, model.__name__
    assert count(engine, M.User, M.User.user_id == a) == 0
    assert count(engine, M.SearchResult) < results_before                        # alice's result rows went with her searches
    assert count(engine, M.SearchResult, M.SearchResult.search_id.in_(select(M.Search.search_id).where(M.Search.user_id == b))) > 0
    assert count(engine, M.Job) == jobs_before and count(engine, M.Skill) == skills_before     # public data is untouched
    assert count(engine, M.UserProfile, M.UserProfile.user_id == b) == 1 and count(engine, M.JobInteraction, M.JobInteraction.user_id == b) == 1 and count(engine, M.UserSkill, M.UserSkill.user_id == b) == 1
    assert bob.get("/dashboard").status_code == 200                              # Bob is unaffected

    assert c.get("/dashboard").status_code == 302                                # the session ended
    fresh = app.test_client()
    assert fresh.post("/login", data={"csrf_token": csrf_of(fresh), "identifier": "alice", "password": PW}).status_code == 401
    assert register(fresh, "alice", "alice@example.com", PW).status_code == 302   # the name can be registered again, as a new empty account


def test_delete_cannot_be_used_to_guess_the_password(site):
    _, _, c, engine = site
    register(c)
    codes = [post(c, "/account/delete", password=f"guess {n}", understood="1").status_code for n in range(12)]
    assert 429 in codes and count(engine, M.User) == 1


# ================================================================================================================================ navigation
def test_account_menu_links_for_logged_in_users(site):
    _, _, c, _ = site
    register(c)
    html = c.get("/").data.decode()
    assert 'href="/account"' in html and 'href="/dashboard"' in html
    nav = html[html.index('class="navbar-nav me-auto"'):]
    assert 'href="/saved"' in nav[: nav.index("</ul>")] and 'href="/profile"' in nav[: nav.index("</ul>")]
    assert re.search(r'class="nav-link active" href="/saved" aria-current="page"', c.get("/saved").data.decode())
    assert re.search(r'class="nav-link active" href="/profile" aria-current="page"', c.get("/profile").data.decode())
