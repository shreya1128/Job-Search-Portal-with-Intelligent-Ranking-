import datetime as dt
import shutil

import numpy as np
import pytest

from jobintel.extraction.extractor import Extractor
from jobintel.extraction.lexicon import SkillLexicon
from jobintel.rec.documents import ScannedResumeOCR, analyze_document, char_error_rate, read_document
from jobintel.rec.resume import ResumeAnalyzer
from jobintel.rec.synthetic_resumes import LABEL, SKILL_POOL, make_resume, render_resume_image

TODAY = dt.date(2026, 10, 12)
needs_ocr = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


@pytest.fixture(scope="module")
def an():
    return ResumeAnalyzer(Extractor(SkillLexicon.from_seed(), cities=["pune", "mumbai"]))


@pytest.fixture(scope="module")
def sample():
    return make_resume(np.random.default_rng(2024), TODAY)


def test_character_error_rate():
    assert char_error_rate("hello world", "hello   world") == 0.0                       # whitespace / line breaks are not errors
    assert char_error_rate("abcd", "abxd") == pytest.approx(0.25)
    assert char_error_rate("abcd", "") == 1.0 and char_error_rate("", "") == 0.0


def test_synthetic_resumes_are_labelled_deterministic_and_consistent():
    t1, tr1 = make_resume(np.random.default_rng(5), TODAY)
    t2, tr2 = make_resume(np.random.default_rng(5), TODAY)
    assert t1 == t2 and tr1 == tr2 and "SYNTHETIC" in tr1.label and LABEL.startswith("SYNTHETIC")
    assert tr1.email in t1 and tr1.name in t1.splitlines()[0] and all(s in t1 for s in tr1.skills)
    assert tr1.experience_months == sum(j["months"] for j in tr1.jobs)                  # job spans are inclusive, so the truth is the sum of the durations
    spans = sorted((j["start"], j["end"]) for j in tr1.jobs)
    assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))                           # jobs do not overlap


def test_the_analyser_recovers_the_synthetic_truth_from_clean_text(an, sample):
    text, t = sample
    p = an.analyze(text, today=TODAY)
    assert p.name == t.name and p.emails == [t.email] and any(t.phone.replace(" ", "") in x.replace(" ", "") for x in p.phones)
    assert set(t.companies) <= set(p.organisations)
    if any(r["counted"] and "/" in r["text"] or any(m in r["text"] for m in ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")) for r in p.date_ranges):
        assert abs(p.computed_experience_years - t.experience_months / 12) < 0.1


def test_docx_text_pdf_and_unsupported(an, sample, tmp_path):
    text, t = sample
    docx = pytest.importorskip("docx")
    d = docx.Document()
    for line in text.split("\n"):
        d.add_paragraph(line)
    tbl = d.add_table(rows=1, cols=2)
    tbl.rows[0].cells[0].text, tbl.rows[0].cells[1].text = "Languages", "English, Hindi"
    f = tmp_path / "cv.docx"
    d.save(f)
    doc = read_document(f)
    assert doc.method == "docx" and t.email in doc.text and "English, Hindi" in doc.text
    assert analyze_document(f, an, today=TODAY)[0].name == t.name
    rl = pytest.importorskip("reportlab.pdfgen.canvas")
    pdf = tmp_path / "cv.pdf"
    c = rl.Canvas(str(pdf))
    y = 800
    for line in text.split("\n"):
        c.drawString(40, y, line)
        y -= 14
    c.save()
    pd_ = read_document(pdf)
    assert pd_.method == "pdf_text" and t.email in pd_.text
    with pytest.raises(ValueError):
        read_document(tmp_path / "cv.xyz")


@needs_ocr
def test_ocr_clean_page_is_near_perfect_and_carries_provenance(an, sample, tmp_path):
    cv2 = pytest.importorskip("cv2")
    text, t = sample
    img, ref = render_resume_image(text, seed=1)
    f = tmp_path / "scan.png"
    cv2.imwrite(str(f), img)
    prof, doc = analyze_document(f, an, ScannedResumeOCR(), today=TODAY)
    assert doc.method == "ocr" and char_error_rate(ref, doc.text) < 0.02 and doc.ocr_confidence > 80
    assert prof.name == t.name and prof.emails == [t.email]
    assert prof.warnings[0].startswith("text obtained by OCR")                         # provenance is never hidden


@needs_ocr
def test_deskew_recovers_a_rotated_scan_and_regions_alone_would_fail(sample):
    text, _ = sample
    img, ref = render_resume_image(text, rotate_deg=3.0, seed=2)
    r = ScannedResumeOCR().read(img)
    assert abs(r.skew_degrees - (-3.0)) < 0.5 and char_error_rate(ref, r.text) < 0.02
    no = ScannedResumeOCR(deskew=False).read(img, layout="regions")
    assert char_error_rate(ref, no.text) > 0.1                                           # why the deskew step exists (measured: 0.33 on the tuning page)


@needs_ocr
def test_two_columns_need_region_detection_and_reading_order(sample):
    text, _ = sample
    img, ref = render_resume_image(text, columns=2, seed=3)
    auto = ScannedResumeOCR().read(img)
    page = ScannedResumeOCR().read(img, layout="page")
    assert len(auto.regions) > 5 and char_error_rate(ref, auto.text) < 0.02
    assert char_error_rate(ref, page.text) > 0.2                                         # whole-page OCR interleaves the two columns
    with pytest.raises(ValueError):
        ScannedResumeOCR().read(img, layout="bogus")


@needs_ocr
def test_scanned_pdf_uses_ocr_and_a_blank_page_is_reported_not_invented(an, sample, tmp_path):
    pytest.importorskip("pypdfium2")
    from PIL import Image
    text, t = sample
    img, _ = render_resume_image(text, seed=4)
    pdf = tmp_path / "scan.pdf"
    Image.fromarray(img).save(pdf, "PDF", resolution=200)
    d = read_document(pdf, ScannedResumeOCR())
    assert d.method == "pdf_ocr" and t.email in d.text and any("scan" in w for w in d.warnings)
    blank = ScannedResumeOCR().read(np.full((1200, 900), 255, np.uint8))
    assert blank.text == "" and any("no text recognised" in w for w in blank.warnings)
