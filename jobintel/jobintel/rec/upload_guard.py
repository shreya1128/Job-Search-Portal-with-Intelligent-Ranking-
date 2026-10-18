"""Upload guard: every resume file passes through here BEFORE any parser sees it.

A resume is untrusted input and the parsers behind it (pdfplumber / pypdf / python-docx / OpenCV / Tesseract) are not written to be handed hostile files.
Enforced (each rule has a test in tests/test_rec_upload_guard.py):
  * extension whitelist, empty file, size limit, content sniffing (the bytes must match the extension)
  * text: no NUL / control-character soup (a binary file named .txt), extracted-text length cap
  * PDF: not encrypted, readable, page count checked with pypdf, at most MAX_PAGES pages are passed on (a shorter copy is written)
  * DOCX: a valid OOXML package, member-count / total-UNCOMPRESSED-size / compression-ratio limits (zip bombs), no absolute or '..' member paths
  * image: header-only decode with Pillow: pixel limit, minimum side, frame limit
  * the parser gets a private temporary file with a constant name; the visitor's file name never touches the disk (only a sanitised display name exists)
  * every parser failure becomes UploadRejected with a generic message (no paths, no library names); the cause is logged on the server without the message text
  * parsing is gated by a bounded semaphore so a few uploads cannot occupy every core
NOT covered here (needs a worker process): a hard wall-clock deadline for OCR; today OCR is bounded by the page cap, the pixel cap and Tesseract's per-call time-out.
"""
from __future__ import annotations

import io
import logging
import re
import tempfile
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger("jobintel.upload")

MAX_BYTES = 15 * 1024 * 1024
MAX_PAGES = 8
MAX_PIXELS = 40_000_000                    # an A4 page at 600 dpi is 35 megapixels
MIN_SIDE = 200
MAX_FRAMES = 8
MAX_TEXT_CHARS = 200_000
MAX_DOCX_UNCOMPRESSED = 30 * 1024 * 1024
MAX_DOCX_MEMBERS = 500
MAX_DOCX_RATIO = 100
MAX_CONCURRENT = 2
BUSY_WAIT_S = 20.0

TEXT = {".txt", ".md"}
IMAGES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ALLOWED = TEXT | IMAGES | {".pdf", ".docx"}
_gate = threading.BoundedSemaphore(MAX_CONCURRENT)


class UploadRejected(ValueError):
    """The file was refused or could not be read; the message is safe to show to the visitor."""


@dataclass
class Checked:
    suffix: str
    data: bytes
    warnings: List[str] = field(default_factory=list)


def _basename(filename) -> str:
    return re.split(r"[\\/]", str(filename or ""))[-1]


def display_name(filename) -> str:
    """what may be shown / stored as the file's name: no directories, no markup characters, at most 80 characters."""
    return re.sub(r"[^\w.\- ()]", "_", _basename(filename))[:80] or "resume"


def _check_text(data: bytes) -> None:
    head = data[:8192]
    if b"\x00" in head or sum(1 for b in head if b < 32 and b not in (9, 10, 13)) > 0.02 * len(head):
        raise UploadRejected("The file does not look like plain text. Save it as UTF-8 text, or upload a PDF or DOCX.")


def _check_pdf(data: bytes) -> Tuple[bytes, List[str]]:
    if not data[:1024].lstrip().startswith(b"%PDF"):
        raise UploadRejected("The file has a .pdf extension but is not a PDF.")
    try:
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(io.BytesIO(data))
        encrypted = bool(reader.is_encrypted)
        n = 0 if encrypted else len(reader.pages)
    except Exception:                                                  # noqa: BLE001 - damaged / truncated
        raise UploadRejected("The PDF could not be opened; it may be damaged.") from None
    if encrypted:
        raise UploadRejected("The PDF is password-protected. Remove the password and upload it again.")
    if n == 0:
        raise UploadRejected("The PDF has no pages.")
    if n <= MAX_PAGES:
        return data, []
    w = PdfWriter()
    for page in reader.pages[:MAX_PAGES]:
        w.add_page(page)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue(), [f"The PDF has {n} pages; only the first {MAX_PAGES} were read."]


def _check_docx(data: bytes) -> None:
    if not data.startswith(b"PK"):
        raise UploadRejected("The file has a .docx extension but is not a Word document.")
    try:
        infos = zipfile.ZipFile(io.BytesIO(data)).infolist()
    except zipfile.BadZipFile:
        raise UploadRejected("The Word document is damaged.") from None
    if len(infos) > MAX_DOCX_MEMBERS:
        raise UploadRejected("The Word document has too many parts to be a normal resume.")
    total = 0
    for i in infos:
        name = i.filename.replace("\\", "/")
        if name.startswith("/") or ".." in name.split("/"):
            raise UploadRejected("The Word document contains unsafe file paths.")
        if i.flag_bits & 0x1:
            raise UploadRejected("The Word document is encrypted.")
        total += i.file_size
        if i.file_size > 1_000_000 and i.file_size > MAX_DOCX_RATIO * max(i.compress_size, 1):
            raise UploadRejected("The Word document expands to far more data than a normal resume.")
    if total > MAX_DOCX_UNCOMPRESSED:
        raise UploadRejected(f"The Word document expands to more than {MAX_DOCX_UNCOMPRESSED // 2**20} MB.")
    names = {i.filename for i in infos}
    if "word/document.xml" not in names or "[Content_Types].xml" not in names:
        raise UploadRejected("The file is not a Word (.docx) document.")


def _check_image(data: bytes, suffix: str) -> None:
    sig = data[:12]
    ok = {".png": sig.startswith(b"\x89PNG\r\n\x1a\n"), ".jpg": sig.startswith(b"\xff\xd8\xff"), ".jpeg": sig.startswith(b"\xff\xd8\xff"), ".bmp": sig.startswith(b"BM"),
          ".tif": sig[:4] in (b"II*\x00", b"MM\x00*"), ".tiff": sig[:4] in (b"II*\x00", b"MM\x00*")}[suffix]
    if not ok:
        raise UploadRejected(f"The file has a {suffix} extension but is not that kind of image.")
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:                       # header only: the pixels are not decoded
            w, h = im.size
            frames = getattr(im, "n_frames", 1)
        if frames == 1 and w * h <= MAX_PIXELS:                        # cheap integrity check (chunk CRCs); a truncated PNG is caught here
            with Image.open(io.BytesIO(data)) as im2:
                im2.verify()
    except Exception:                                                  # noqa: BLE001 - damaged, or Pillow's own decompression-bomb error
        raise UploadRejected("The image could not be read; it may be damaged or far too large.") from None
    if min(w, h) < MIN_SIDE:
        raise UploadRejected(f"The image is too small to contain a readable page (at least {MIN_SIDE} px on each side).")
    if w * h > MAX_PIXELS:
        raise UploadRejected(f"The image is too large ({w} x {h} px; at most {MAX_PIXELS // 1_000_000} megapixels).")
    if frames > MAX_FRAMES:
        raise UploadRejected(f"The image has {frames} pages; at most {MAX_FRAMES} are allowed.")


def inspect(data: bytes, filename: str) -> Checked:
    """validate the bytes against their extension; returns the (possibly shortened) bytes to hand to a parser."""
    suffix = Path(_basename(filename)).suffix.lower()
    if suffix not in ALLOWED:
        raise UploadRejected(f"{suffix or 'This file type'} is not supported. Use PDF, DOCX, TXT or an image (PNG, JPEG, TIFF, BMP)." + (" Convert .doc files to .docx or PDF." if suffix == ".doc" else ""))
    if not data:
        raise UploadRejected("The file is empty.")
    if len(data) > MAX_BYTES:
        raise UploadRejected(f"The file is larger than {MAX_BYTES // 2**20} MB.")
    warnings: List[str] = []
    if suffix in TEXT:
        _check_text(data)
    elif suffix == ".pdf":
        data, warnings = _check_pdf(data)
    elif suffix == ".docx":
        _check_docx(data)
    else:
        _check_image(data, suffix)
    return Checked(suffix, data, warnings)


def read_resume(data: bytes, filename: str, *, reader=None):
    """guard + parse.  ``reader`` is a module with ``read_document(path) -> DocumentText`` (default: ``jobintel.rec.documents``)."""
    checked = inspect(data, filename)
    if reader is None:
        from . import documents as reader
    if not _gate.acquire(timeout=BUSY_WAIT_S):
        raise UploadRejected("The server is busy reading other resumes. Please try again in a minute.")
    try:
        with tempfile.TemporaryDirectory(prefix="jobintel_resume_") as tmp:      # private (0700) directory; constant file name
            path = Path(tmp) / f"upload{checked.suffix}"
            path.write_bytes(checked.data)
            try:
                doc = reader.read_document(path)
            except Exception as e:                                       # noqa: BLE001
                if type(e).__name__ == "DocumentError":                  # the reader's own messages are written for visitors
                    raise UploadRejected(str(e)) from None
                if isinstance(e, RuntimeError):                          # a missing OCR program / package is the operator's problem: say so in the log
                    log.error("resume parser unavailable: %s", e)
                else:
                    log.warning("resume parser failed: %s", type(e).__name__)
                raise UploadRejected("This file could not be read. Try saving it as a text-based PDF or DOCX.") from None
    finally:
        _gate.release()
    if len(doc.text) > MAX_TEXT_CHARS:
        doc.text = doc.text[:MAX_TEXT_CHARS]
        checked.warnings.append(f"The text was cut to {MAX_TEXT_CHARS:,} characters.")
    doc.warnings = list(getattr(doc, "warnings", []) or []) + checked.warnings
    return doc
