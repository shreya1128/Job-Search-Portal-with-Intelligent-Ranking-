"""Resume INPUT layer: plain text, DOCX, PDF (text layer or scanned) and images (scans / photos) -> text for the ResumeAnalyzer.

Scanned-document pipeline (classical image processing + Tesseract OCR; no trained layout model):
  1 load -> grayscale -> rescale so the page is ~2000 px wide (OCR works best near 300 dpi) -> median denoise
  2 deskew: the angle (-10..+10 deg) that maximises the variance of the row sums of the binarised page (text lines become horizontal), then rotate
  3 binarise (Otsu, or adaptive thresholding when the background is uneven)
  4 REGION detection: connected components -> median character height -> morphological dilation merges words / lines into text blocks -> bounding boxes
  5 READING ORDER: blocks narrower than 60% of the page are assigned to the left / right column; full-width blocks separate bands; each band is read left column then right column, top to bottom
  6 OCR of every block (Tesseract, per-word confidences) -> text + mean confidence; low confidence is reported as a warning, never hidden.
Everything is measured on SYNTHETIC rendered resumes (scripts/evaluate_resume_pipeline.py); no real scans were available, so real-world accuracy is unknown.
OCR needs the `tesseract` binary + pytesseract + opencv; PDF text needs pypdf; DOCX needs python-docx; scanned PDFs need PyMuPDF.  Missing pieces raise a clear error, never silently return nothing."""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
TEXT_EXT = {".txt", ".md", ""}


def normalise_ws(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def char_error_rate(ref: str, hyp: str) -> float:
    """Levenshtein distance / len(reference) on whitespace-normalised strings (line breaks do not count as errors)."""
    a, b = normalise_ws(ref), normalise_ws(hyp)
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(a)


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    text: str = ""
    confidence: Optional[float] = None


@dataclass
class OCRResult:
    text: str
    regions: List[Region]
    skew_degrees: float
    mean_confidence: Optional[float]
    n_words: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class DocumentText:
    text: str
    method: str                                   # 'text' | 'docx' | 'pdf_text' | 'ocr' | 'pdf_ocr'
    pages: int = 1
    ocr_confidence: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


def _need(module: str, what: str):
    try:
        return __import__(module)
    except ImportError as e:
        raise RuntimeError(f"{what} needs the python package '{module}' (pip install {module})") from e


class ScannedResumeOCR:
    def __init__(self, lang: str = "eng", target_width: int = 2000, deskew: bool = True, denoise: bool = True, psm: int = 6, min_confidence: float = 55.0, ocr_on: str = "gray"):
        if shutil.which("tesseract") is None:
            raise RuntimeError("OCR needs the 'tesseract' program on PATH (e.g. apt install tesseract-ocr)")
        self.cv2, self.pt = _need("cv2", "OCR"), _need("pytesseract", "OCR")
        self.lang, self.target_width, self.deskew, self.denoise, self.psm, self.min_conf, self.ocr_on = lang, target_width, deskew, denoise, psm, min_confidence, ocr_on

    # --------------------------------------------------------------------------------------------------------------- loading / preprocessing
    def _gray(self, image) -> np.ndarray:
        cv2 = self.cv2
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"cannot read image {image}")
            return img
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return arr.astype(np.uint8)

    def estimate_skew(self, gray: np.ndarray, max_deg: float = 10.0) -> float:
        cv2 = self.cv2
        s = 800.0 / gray.shape[1]
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
        bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        h, w = bw.shape
        c = (w / 2, h / 2)

        def score(a: float) -> float:
            M = cv2.getRotationMatrix2D(c, a, 1.0)
            return float(np.var(cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST).sum(axis=1)))
        coarse = np.arange(-max_deg, max_deg + 1e-9, 0.5)
        best = float(coarse[int(np.argmax([score(a) for a in coarse]))])
        fine = np.arange(best - 0.5, best + 0.5 + 1e-9, 0.1)
        return float(fine[int(np.argmax([score(a) for a in fine]))])

    def preprocess(self, image) -> Tuple[np.ndarray, np.ndarray, float]:
        """returns (deskewed gray, binary with black text on white, skew angle that was corrected, degrees)."""
        cv2 = self.cv2
        g = self._gray(image)
        if g.shape[1] < 1600 or g.shape[1] > 3200:
            s = self.target_width / g.shape[1]
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        if self.denoise:
            g = cv2.medianBlur(g, 3)
        skew = self.estimate_skew(g) if self.deskew else 0.0
        if abs(skew) >= 0.2:
            h, w = g.shape
            g = cv2.warpAffine(g, cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0), (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        uneven = float(np.std(cv2.GaussianBlur(g, (0, 0), 30))) > 12.0
        if uneven:
            bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)
        else:
            bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return g, bw, skew

    # --------------------------------------------------------------------------------------------------------------- regions / reading order
    def detect_regions(self, bw: np.ndarray) -> List[Region]:
        cv2 = self.cv2
        inv = 255 - bw
        n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        hs = stats[1:, cv2.CC_STAT_HEIGHT]
        hs = hs[(hs > 6) & (hs < 120)]
        if not len(hs):
            return []
        mh = float(np.median(hs))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(1.5 * mh)), max(3, int(0.8 * mh))))
        cnts, _ = cv2.findContours(cv2.dilate(inv, kernel), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h >= 2 * mh * mh and h >= 0.6 * mh:
                boxes.append(Region(x, y, w, h))
        return boxes

    @staticmethod
    def reading_order(regions: List[Region], page_width: int) -> List[Region]:
        full = lambda r: r.w > 0.6 * page_width
        ordered, group = [], []

        def flush():
            left = sorted((r for r in group if r.x + r.w / 2 < page_width / 2), key=lambda r: r.y)
            right = sorted((r for r in group if r.x + r.w / 2 >= page_width / 2), key=lambda r: r.y)
            ordered.extend(left + right)
            group.clear()
        for r in sorted(regions, key=lambda r: r.y):
            if full(r):
                flush()
                ordered.append(r)
            else:
                group.append(r)
        flush()
        return ordered

    # --------------------------------------------------------------------------------------------------------------- OCR
    def _ocr_crop(self, crop: np.ndarray) -> Tuple[str, Optional[float], int]:
        cv2 = self.cv2
        crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        d = self.pt.image_to_data(crop, lang=self.lang, config=f"--psm {self.psm}", output_type=self.pt.Output.DICT)
        lines, confs, key, cur = [], [], None, []
        for t, c, b, p, l in zip(d["text"], d["conf"], d["block_num"], d["par_num"], d["line_num"]):
            if not str(t).strip() or float(c) < 0:
                continue
            k = (b, p, l)
            if k != key and cur:
                lines.append(" ".join(cur))
                cur = []
            key = k
            cur.append(str(t))
            confs.append(float(c))
        if cur:
            lines.append(" ".join(cur))
        return "\n".join(lines), (float(np.mean(confs)) if confs else None), len(confs)

    @staticmethod
    def is_multicolumn(regions: List[Region], page_width: int) -> bool:
        narrow = [r for r in regions if r.w <= 0.6 * page_width]
        return sum(r.x + r.w / 2 < page_width / 2 for r in narrow) >= 2 and sum(r.x + r.w / 2 >= page_width / 2 for r in narrow) >= 2

    def read(self, image, layout: str = "auto") -> OCRResult:
        """layout: 'auto' = region OCR with column-aware reading order only when a multi-column page is detected (whole-page OCR is more accurate on single-column
        pages), 'regions' = always region OCR, 'page' = always whole-page OCR."""
        if layout not in ("auto", "regions", "page"):
            raise ValueError("layout must be 'auto', 'regions' or 'page'")
        g, bw, skew = self.preprocess(image)
        src = bw if self.ocr_on == "binary" else g
        warnings: List[str] = []
        regions: List[Region] = []
        if layout != "page":
            regions = self.reading_order(self.detect_regions(bw), bw.shape[1])
            if layout == "auto" and not self.is_multicolumn(regions, bw.shape[1]):
                regions = []
        if not regions:
            if layout == "regions":
                warnings.append("no text region found: OCR of the whole page")
            text, conf, n = self._ocr_crop(src)
            regions = [Region(0, 0, bw.shape[1], bw.shape[0], text, conf)]
        else:
            n = 0
            for r in regions:
                r.text, r.confidence, k = self._ocr_crop(src[r.y: r.y + r.h, r.x: r.x + r.w])
                n += k
        text = "\n\n".join(r.text for r in regions if r.text)
        confs = [r.confidence for r in regions if r.confidence is not None]
        wts = [max(len(r.text.split()), 1) for r in regions if r.confidence is not None]
        mean = float(np.average(confs, weights=wts)) if confs else None
        if mean is None:
            warnings.append("no text recognised")
        elif mean < self.min_conf:
            warnings.append(f"low OCR confidence ({mean:.0f} < {self.min_conf:.0f}): extracted fields may be wrong - check the document")
        return OCRResult(text, regions, skew, mean, n, warnings)


# ------------------------------------------------------------------------------------------------------------------- readers
def read_docx(path: Path) -> DocumentText:
    docx = _need("docx", "DOCX input")
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            cells = []
            for c in row.cells:
                if c.text.strip() and c.text.strip() not in cells:
                    cells.append(c.text.strip())
            parts.append(" | ".join(cells))
    return DocumentText("\n".join(parts), "docx", pages=1)


def read_pdf(path: Path, ocr: Optional[ScannedResumeOCR] = None, min_chars_per_page: int = 40, dpi: int = 250,
             max_ocr_pages: int = 5, ocr_timeout_s: float = 15.0) -> DocumentText:
    pypdf = _need("pypdf", "PDF input")
    reader = pypdf.PdfReader(str(path))
    texts = [(p.extract_text() or "") for p in reader.pages]
    n = len(texts)
    if sum(len(t.strip()) for t in texts) >= min_chars_per_page * max(n, 1):
        return DocumentText("\n".join(texts), "pdf_text", pages=n)
    warn = ["no usable text layer: the PDF is a scan, OCR was used"]
    ocr = ocr or ScannedResumeOCR()
    out, confs = [], []
    import time
    t0 = time.time()
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(path))
        page_count = len(doc)
        for i, page in enumerate(doc):
            if i >= max_ocr_pages:
                warn.append(f"OCR budget reached: only the first {max_ocr_pages} of {page_count} scanned pages were processed.")
                break
            if time.time() - t0 > ocr_timeout_s:
                warn.append(f"OCR time budget exceeded ({ocr_timeout_s:.0f}s): remaining scanned pages skipped.")
                break
            pix = page.render(scale=dpi / 72.0).to_pil()
            img = np.array(pix)
            res = ocr.read(img[:, :, :3] if img.ndim == 3 and img.shape[2] >= 3 else img)
            out.append(res.text)
            if res.mean_confidence is not None:
                confs.append(res.mean_confidence)
            warn += res.warnings
        return DocumentText("\n\n".join(out), "pdf_ocr", pages=min(page_count, max_ocr_pages),
                            ocr_confidence=float(np.mean(confs)) if confs else None,
                            warnings=list(dict.fromkeys(warn)))
    except ImportError:
        return DocumentText("\n".join(texts), "pdf_text", pages=n,
                            warnings=["scanned PDF OCR requires pypdfium2 for page rendering"])
    except Exception as e:
        return DocumentText("\n".join(texts), "pdf_text", pages=n,
                            warnings=[f"scanned PDF OCR rendering failed: {e}"])


def read_document(path, ocr: Optional[ScannedResumeOCR] = None) -> DocumentText:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return DocumentText(path.read_text(encoding="utf-8", errors="replace"), "text")
    if ext == ".docx":
        return read_docx(path)
    if ext == ".pdf":
        return read_pdf(path, ocr)
    if ext in IMAGE_EXT:
        r = (ocr or ScannedResumeOCR()).read(path)
        return DocumentText(r.text, "ocr", ocr_confidence=r.mean_confidence, warnings=r.warnings)
    raise ValueError(f"unsupported resume format '{ext}' (supported: text, .docx, .pdf, {', '.join(sorted(IMAGE_EXT))})")


def analyze_document(path, analyzer, ocr: Optional[ScannedResumeOCR] = None, **kw):
    """read_document + ResumeAnalyzer; OCR provenance and warnings are carried into the profile's warnings."""
    doc = read_document(path, ocr)
    prof = analyzer.analyze(doc.text, **kw)
    if doc.method in ("ocr", "pdf_ocr"):
        prof.warnings.insert(0, f"text obtained by OCR" + (f" (mean confidence {doc.ocr_confidence:.0f})" if doc.ocr_confidence is not None else "") + ": verify the extracted fields")
    prof.warnings += doc.warnings
    return prof, doc
