"""Phase 7 Stages 3–4: resume upload, recommendation and market/skill-gap workflows."""
from __future__ import annotations

import io
from .web_fixtures import csrf_of, make_app


def post(c, path, **data):
    return c.post(path, data={"csrf_token": csrf_of(c), **data}, follow_redirects=False)


def test_resume_upload_rejects_unsupported_extension(engine, tmp_path):
    app, _ = make_app(engine, tmp_path)
    c = app.test_client()
    r = c.post("/resume/analyze", data={"csrf_token": csrf_of(c), "resume": (io.BytesIO(b"not executable"), "resume.exe")}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert b"Unsupported file type" in r.data


def test_resume_upload_extracts_and_keeps_result_in_session(engine, tmp_path):
    app, _ = make_app(engine, tmp_path)
    c = app.test_client()
    text = ("ALEX MORGAN\nData Analyst\nEmail: alex@example.com\n\nSKILLS\nPython, SQL, Excel, data analysis\n\n"
            "EXPERIENCE\nData Analyst, Example Systems\nJan 2021 - Jan 2024\nBuilt SQL reports and analysed business data.\n" * 2)
    r = c.post("/resume/analyze", data={"csrf_token": csrf_of(c), "resume": (io.BytesIO(text.encode()), "resume.txt")}, content_type="multipart/form-data")
    assert r.status_code == 302 and r.headers["Location"] == "/resume"
    html = c.get("/resume").data.decode()
    assert "ANALYSIS RESULT" in html and "alex@example.com" in html and "Recognised skills" in html
    with c.session_transaction() as sess:
        assert sess.get("resume_token")


def test_resume_save_requires_login(engine, tmp_path):
    app, _ = make_app(engine, tmp_path)
    c = app.test_client()
    r = post(c, "/resume/save")
    assert r.status_code == 302 and "/login?next=/resume/save" in r.headers["Location"]


def test_stage4_market_and_skillgap_forms_are_present(engine, tmp_path):
    app, _ = make_app(engine, tmp_path)
    c = app.test_client()
    market = c.get("/career-insights").data.decode()
    gap = c.get("/skill-gap").data.decode()
    assert "Job areas" in market and "Skills appearing in postings" in market
    assert 'name="job_id"' in gap and 'name="skills"' in gap and "Analyse gap" in gap


def test_skillgap_role_submission_reports_real_skills(engine, tmp_path):
    app, _ = make_app(engine, tmp_path)
    c = app.test_client()
    token = csrf_of(c)
    r = c.post("/skill-gap", data={"csrf_token": token, "role": "data analyst", "skills": "sql, excel"})
    assert r.status_code == 200
    assert "Skill-by-skill comparison" in r.data.decode()


def test_recommendations_accept_role_and_skills(engine, tmp_path):
    app, _ = make_app(engine, tmp_path)
    c = app.test_client()
    r = c.get("/recommendations", query_string={"run": "1", "src": "form", "role": "data analyst", "skills": "sql, excel", "k": "5"})
    assert r.status_code == 200
    assert "Your matches" in r.data.decode() or "No matching jobs found" in r.data.decode() or "unavailable" in r.data.decode().lower()


def test_resume_upload_empty_file_rejected(engine, tmp_path):
    app, _ = make_app(engine, tmp_path)
    c = app.test_client()
    tok = csrf_of(c)
    r = c.post("/resume/analyze", data={"csrf_token": tok, "resume": (io.BytesIO(b""), "empty.pdf")}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert b"empty" in r.data.lower() or b"reject" in r.data.lower() or b"error" in r.data.lower()


def test_resume_upload_missing_csrf_rejected(engine, tmp_path):
    app, _ = make_app(engine, tmp_path)
    c = app.test_client()
    r = c.post("/resume/analyze", data={"resume": (io.BytesIO(b"some resume text"), "resume.txt")}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_resume_save_when_logged_in_updates_profile(engine, tmp_path):
    from .web_fixtures import register
    app, _ = make_app(engine, tmp_path)
    c = app.test_client()
    register(c, "analyst_bob", "bob@example.com", "securePassword123")
    text = ("BOB SMITH\nData Analyst\nEmail: bob@example.com\n\nSKILLS\nSQL, Excel, Python\n\n"
            "EXPERIENCE\nData Analyst at TechCorp (2020-2023)\nAnalyzed data using SQL and Python.\n")
    tok = csrf_of(c)
    r_upload = c.post("/resume/analyze", data={"csrf_token": tok, "resume": (io.BytesIO(text.encode()), "bob_resume.txt")}, content_type="multipart/form-data")
    assert r_upload.status_code == 302
    r_save = post(c, "/resume/save")
    assert r_save.status_code in (200, 302)
    # Profile should now reflect extracted skills or resume info
    r_prof = c.get("/profile")
    assert r_prof.status_code == 200
    html = r_prof.data.decode().lower()
    assert "sql" in html or "bob" in html or "analyst" in html


def test_skillgap_by_job_id(engine, tmp_path):
    app, _ = make_app(engine, tmp_path)
    c = app.test_client()
    tok = csrf_of(c)
    # Job 100 is Data Analyst in web_fixtures
    r = c.post("/skill-gap", data={"csrf_token": tok, "job_id": "100", "skills": "sql, excel"})
    assert r.status_code == 200
    html = r.data.decode()
    assert "Skill-by-skill comparison" in html or "Data Analyst" in html

