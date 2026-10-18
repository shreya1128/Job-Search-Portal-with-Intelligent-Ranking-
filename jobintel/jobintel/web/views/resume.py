"""Resume analysis UI for Phase 7 Stage 3.

Files are parsed in a temporary directory and removed immediately. An analysed resume is kept
in the server-side ephemeral store for this browser session; it is persisted only after the user
explicitly chooses Save to profile.
"""
from __future__ import annotations

import logging
import re
import tempfile
import zipfile
from pathlib import Path

from flask import Blueprint, abort, flash, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from ...rec.documents import read_document
from ..auth import load_current_user, login_required
from ..backend import Unavailable
from ._common import backend, db_session, ext, public_reason
from ._profile import RESUME_KEY, session_resume

bp = Blueprint("resume", __name__)
log = logging.getLogger("jobintel.web")
ALLOWED = {".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_TEXT = 120_000
MAX_PDF_PAGES = 20


def _validate_upload(path: Path, suffix: str) -> None:
    """Reject extension/content mismatches and structurally unsafe or oversized documents."""
    head = path.read_bytes()[:16]
    if suffix in {".txt", ".md"}:
        raw = path.read_bytes()
        if b"\x00" in raw[:4096]:
            raise ValueError("This file does not look like plain text. Upload a text resume, PDF, Word document or image.")
        if len(raw) > MAX_TEXT * 4:
            raise ValueError("The text file is too large to analyse safely.")
        try:
            raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError("The text file must use UTF-8 encoding.")
    elif suffix == ".pdf" and not head.startswith(b"%PDF-"):
        raise ValueError("The file extension says PDF, but the file is not a valid PDF.")
    elif suffix == ".docx":
        if not zipfile.is_zipfile(path):
            raise ValueError("The Word document is invalid or damaged.")
        try:
            with zipfile.ZipFile(path) as z:
                names = set(z.namelist())
                if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                    raise ValueError("The Word document is missing its document content.")
                total = sum(x.file_size for x in z.infolist())
                if total > 30 * 1024 * 1024 or any(x.file_size > 10 * 1024 * 1024 for x in z.infolist()):
                    raise ValueError("The Word document expands beyond the safe processing limit.")
        except zipfile.BadZipFile:
            raise ValueError("The Word document is invalid or damaged.")
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        try:
            from PIL import Image
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                if img.width * img.height > 25_000_000:
                    raise ValueError("This image is too large to process. Use an image below 25 megapixels.")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError("The image is invalid or damaged.") from e
    else:
        raise ValueError("That file type is not supported. Upload PDF, DOCX, TXT, Markdown or a common image format.")


def _profile_summary(profile):
    return {
        "name": profile.name,
        "emails": profile.emails,
        "phones": profile.phones,
        "urls": profile.urls,
        "skills": profile.skills,
        "experience_years": profile.experience_years,
        "experience_source": profile.experience_source,
        "education": profile.best_education,
        "education_items": profile.education,
        "certifications": profile.certifications,
        "roles": profile.roles,
        "locations": profile.locations,
        "organisations": profile.organisations,
        "sections": profile.sections,
        "entities": profile.entities,
        "warnings": profile.warnings,
        "n_words": profile.n_words,
    }


@bp.route("/resume", methods=["GET"])
def analyzer():
    saved = session_resume()
    return render_template("resume.html", result=_profile_summary(saved["profile"]) if saved else None,
                           filename=saved["filename"] if saved else None,
                           method=saved["document"].method if saved else None,
                           pages=saved["document"].pages if saved else None,
                           errors=[], upload_limit=ext().settings.max_upload_mb,
                           current_user=load_current_user())


from ...rec.upload_guard import UploadRejected, display_name, read_resume

@bp.route("/resume/analyze", methods=["POST"])
def analyze():
    f = request.files.get("resume")
    if f is None or not f.filename:
        return render_template("resume.html", result=None, filename=None, method=None, pages=None,
                               errors=["Choose a resume file first."], upload_limit=ext().settings.max_upload_mb,
                               current_user=load_current_user()), 400
    original = secure_filename(f.filename)
    suffix = Path(original).suffix.lower()
    if not original or suffix not in ALLOWED:
        return render_template("resume.html", result=None, filename=None, method=None, pages=None,
                               errors=["Unsupported file type. Use PDF, DOCX, TXT, Markdown or a common image format."],
                               upload_limit=ext().settings.max_upload_mb, current_user=load_current_user()), 400
    try:
        raw_data = f.read()
        if not raw_data:
            raise UploadRejected("The uploaded file is empty.")
        if len(raw_data) > ext().settings.max_upload_mb * 1024 * 1024:
            raise UploadRejected(f"The file exceeds the {ext().settings.max_upload_mb} MB upload limit.")
        
        try:
            analyzer = backend().analyzer
        except Unavailable as e:
            flash(public_reason(e, "Resume analysis"), "error")
            return redirect(url_for("resume.analyzer"))

        # upload_guard validates extension vs content, enforces decompression limits, concurrency, and OCR budgets
        doc = read_resume(raw_data, original)
        if len(doc.text) > MAX_TEXT:
            raise UploadRejected("The extracted text is too long to analyse safely (limit: 120,000 characters).")
        if len(doc.text.strip()) < 20:
            raise UploadRejected("Very little text could be read from this file. Try a clearer scan or a text-based PDF.")

        profile = analyzer.analyze(doc.text)
        profile.warnings.extend(w for w in doc.warnings if w not in profile.warnings)
        safe_name = display_name(original)
        token = ext().store.put({"profile": profile, "document": doc, "filename": safe_name, "text": doc.text})
        session[RESUME_KEY] = token
    except (UploadRejected, ValueError) as e:
        return render_template("resume.html", result=None, filename=original, method=None, pages=None,
                               errors=[str(e)], upload_limit=ext().settings.max_upload_mb, current_user=load_current_user()), 400
    except (IOError, OSError) as e:
        log.warning("resume file system error: %s", e)
        return render_template("resume.html", result=None, filename=original, method=None, pages=None,
                               errors=["Storage or reading error while handling this resume. Please try again."],
                               upload_limit=ext().settings.max_upload_mb, current_user=load_current_user()), 400
    except Exception as e:
        log.exception("unexpected resume analysis error: %s", type(e).__name__)
        return render_template("resume.html", result=None, filename=original, method=None, pages=None,
                               errors=["Jobintel could not read this file. Check that it is a valid document and try a text-based PDF or DOCX."],
                               upload_limit=ext().settings.max_upload_mb, current_user=load_current_user()), 400
    flash("Resume analysed. Review the extracted fields before using them.", "info")
    return redirect(url_for("resume.analyzer"))


@bp.route("/resume/save", methods=["POST"])
@login_required
def save_to_profile():
    user = load_current_user()
    data = session_resume()
    if not data:
        flash("Analyse a resume in this browser session before saving it.", "error")
        return redirect(url_for("resume.analyzer"))
    from ...rec import store as rec_store
    from ...rec.profile import UserProfile
    profile = UserProfile.from_resume(data["profile"])
    try:
        with db_session() as s:
            out = rec_store.save_profile(s, user.user_id, profile, data["profile"], resume_text=data["text"], filename=data["filename"])
            s.commit()
    except Exception as e:
        log.exception("could not save analysed resume for user %s", user.user_id)
        flash("The resume was analysed, but could not be saved to your profile. Your existing profile was not confirmed as updated.", "error")
        return redirect(url_for("resume.analyzer"))
    unknown = out.get("not_in_skills_table", [])
    msg = f"Resume and {out.get('saved_skills', 0)} recognised skills saved to your profile."
    if unknown:
        msg += f" {len(unknown)} extracted skill(s) are not in the current skills table and were not stored."
    flash(msg, "info")
    return redirect(url_for("account.profile"))


@bp.route("/resume/clear", methods=["POST"])
def clear_analysis():
    token = session.pop(RESUME_KEY, None)
    if token:
        ext().store.drop(token)
    flash("The analysed resume was cleared from this browser session.", "info")
    return redirect(url_for("resume.analyzer"))
