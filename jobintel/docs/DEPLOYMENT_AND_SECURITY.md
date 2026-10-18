# JobIntel - Deployment and Security Hardening Guide

This document describes the production deployment architecture, HTTPS termination, secret management, and operational security controls for the JobIntel platform.

---

## 1. Production Architecture Overview

JobIntel follows the standard multi-tier production web deployment model:

```
[ Internet Clients (HTTPS / 443) ]
               │
               ▼
   [ Reverse Proxy (Nginx / Caddy) ]
        ├── TLS / HTTPS Termination (Let's Encrypt)
        ├── Static Asset Caching (/static)
        ├── Rate Limiting & DoS Shielding
        └── Request Header Sanitization
               │
               ▼ (HTTP / 127.0.0.1:5000)
    [ WSGI Application Server ]
        ├── Windows / Cross-platform: Waitress (Multi-threaded)
        └── Linux: Gunicorn (Pre-fork worker pool)
               │
               ▼
         [ Flask App (wsgi.py) ]
         ├── Security Middleware (CSP, CSRF, No-Sniff, X-Frame-Options)
         ├── Upload Guard & Budgets (pypdfium2 / OCR page & time budgets)
         └── In-Memory Search Engine & Hybrid Recommenders
               │
               ▼
     [ Database Storage ]
     ├── MySQL 8.0 / MariaDB (utf8mb4, connection pool)
     └── SQLite (Local development / tests with PRAGMA foreign_keys=ON)
```

---

## 2. Cryptographic Secret Management (SECRET_KEY)

In development, the application falls back to a default testing secret. In production, a cryptographically strong 256-bit random key must be specified.

### Generating a Secure Key
Generate a secure hexadecimal key using Python's standard `secrets` library:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### Configuring `.env`
Set the generated key in your `.env` file (ensure `.env` file permissions are restricted to `600` on Linux, and never committed to source control):

```env
SECRET_KEY=9f8c4b2a8d1e3c5f70123456789abcdef0123456789abcdef0123456789abcdef
```

If `create_app()` detects an unconfigured or default key when running outside `testing` mode, a security warning is logged immediately.

---

## 3. WSGI Production Server Execution

Flask's built-in development server (`run_web.py`) is single-threaded and not designed for concurrent production traffic. Use **Waitress** (Windows/Linux) or **Gunicorn** (Linux).

### Running with Waitress (Native Windows & Cross-Platform)
Waitress is included in `requirements.txt` and supports multi-threaded asynchronous I/O natively on Windows:

```bash
# Start production server with 4 worker threads on localhost:5000:
python scripts/run_production.py --host 127.0.0.1 --port 5000 --threads 4

# Or using the waitress-serve CLI:
waitress-serve --listen=127.0.0.1:5000 wsgi:application
```

### Running with Gunicorn (Linux / Docker)
For Linux production deployments or containerised environments:

```bash
gunicorn -w 4 -b 127.0.0.1:5000 --timeout 60 --access-logfile - --error-logfile - "wsgi:application"
```

---

## 4. HTTPS & Reverse Proxy Configuration

Never expose the WSGI application server directly to the public internet. Use Nginx or Caddy to terminate TLS.

### Nginx Configuration (`/etc/nginx/sites-available/jobintel`)

```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name jobintel.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name jobintel.example.com;

    # SSL Certificates (managed via Certbot)
    ssl_certificate /etc/letsencrypt/live/jobintel.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/jobintel.example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # HSTS (Strict-Transport-Security)
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

    # Maximum file upload size matching application limit (16MB)
    client_max_body_size 16M;

    # Serve static assets directly with caching
    location /static/ {
        alias /path/to/jobintel/jobintel/web/static/;
        expires 30d;
        add_header Cache-Control "public, no-transform";
    }

    # Proxy all other traffic to WSGI application
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### Automated TLS Certificates via Certbot
```bash
sudo certbot --nginx -d jobintel.example.com
```

---

## 5. Security Controls Implemented in Application

1. **Content Security Policy (CSP)**:
   Restricts executable scripts, fonts, and stylesheets to `'self'`, preventing XSS attacks:
   `default-src 'self'; frame-ancestors 'none'; object-src 'none'; ...`
2. **Clickjacking Defense**:
   `X-Frame-Options: DENY` and CSP `frame-ancestors 'none'` prevent embedding in malicious iframes.
3. **MIME Sniffing Prevention**:
   `X-Content-Type-Options: nosniff` disables content-type guessing.
4. **Referrer Policy**:
   `Referrer-Policy: same-origin` shields private query strings and route paths.
5. **CSRF Protection**:
   All state-modifying requests (`POST`, `PUT`, `DELETE`) require a cryptographically signed HMAC token (`csrf_token`).
6. **Upload Hardening (`upload_guard.py`)**:
   - Magic number validation (strictly PDF, DOCX, TXT; forbids executables, macros, ZIP bombs).
   - Maximum file size (10 MB).
   - Maximum OCR pages budget (5 pages).
   - OCR timeout budget (15.0 seconds).
   - Removal of AGPL dependencies (`pypdfium2` + `pypdf` without `fitz`).
7. **Session Isolation**:
   - `SESSION_COOKIE_HTTPONLY=True` prevents JavaScript access to session tokens.
   - `SESSION_COOKIE_SAMESITE="Lax"` mitigates cross-site request forgery.
   - Authenticated responses emit `Cache-Control: no-store`.
