"""Phase 7 Stage 5: Security hardening, headers, CSRF enforcement, session boundaries, and error page integrity."""
from __future__ import annotations

import io
from .web_fixtures import csrf_of, make_app, register


def test_security_headers_present_on_all_responses(engine, tmp_path):
    """Verify that every HTTP response includes strict defense-in-depth headers."""
    app, _ = make_app(engine, tmp_path)
    c = app.test_client()
    for path in ["/", "/jobs", "/recommendations", "/resume", "/career-insights", "/skill-gap", "/about", "/login"]:
        r = c.get(path)
        assert r.status_code in (200, 302), f"Failed for {path}"
        assert r.headers.get("X-Content-Type-Options") == "nosniff"
        assert r.headers.get("X-Frame-Options") in ("DENY", "SAMEORIGIN")
        assert "Content-Security-Policy" in r.headers
        assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]
        assert r.headers.get("Referrer-Policy") in ("same-origin", "strict-origin-when-cross-origin")


def test_csrf_token_required_on_all_modifying_routes(engine, tmp_path):
    """Verify state-changing POST endpoints abort with 400 when CSRF is absent."""
    app, _ = make_app(engine, tmp_path)
    c = app.test_client()

    routes = [
        ("/login", {"username": "foo", "password": "bar"}),
        ("/register", {"username": "foo", "email": "foo@example.com", "password": "bar", "confirm": "bar"}),
        ("/resume/analyze", {"resume": (io.BytesIO(b"text"), "resume.txt")}),
        ("/skill-gap", {"role": "data analyst", "skills": "python"}),
    ]

    for path, data in routes:
        r = c.post(path, data=data, follow_redirects=False)
        assert r.status_code == 400, f"Expected 400 without CSRF on {path}, got {r.status_code}"


def test_authenticated_cache_control_header(engine, tmp_path):
    """Authenticated user sessions must have Cache-Control: no-store to avoid proxy caching."""
    app, _ = make_app(engine, tmp_path)
    c = app.test_client()
    register(c, "cache_tester", "cache@example.com", "mySecurePass123")
    r = c.get("/profile")
    assert r.status_code == 200
    assert r.headers.get("Cache-Control") == "no-store"


def test_custom_404_error_page_does_not_leak_stacktrace(engine, tmp_path):
    """404 error page returns cleanly without server internals or raw exception tracebacks."""
    app, _ = make_app(engine, tmp_path)
    c = app.test_client()
    r = c.get("/this-route-does-not-exist-at-all-12345")
    assert r.status_code == 404
    html = r.data.decode()
    assert "Page not found" in html
    assert "Traceback" not in html
    assert "Werkzeug" not in html


def test_user_session_isolation(engine, tmp_path):
    """User A cannot access or modify User B's profile or saved items."""
    app, _ = make_app(engine, tmp_path)
    client_a = app.test_client()
    client_b = app.test_client()

    register(client_a, "user_alpha", "alpha@example.com", "alphaPass1234")
    register(client_b, "user_beta", "beta@example.com", "betaPass1234")

    # Alpha updates profile
    tok_a = csrf_of(client_a)
    client_a.post("/profile", data={"csrf_token": tok_a, "full_name": "Alice Alpha", "headline": "Alpha Lead"})

    # Beta checks own profile
    r_beta = client_b.get("/profile")
    html_b = r_beta.data.decode()
    assert "Alice Alpha" not in html_b
    assert "Alpha Lead" not in html_b
