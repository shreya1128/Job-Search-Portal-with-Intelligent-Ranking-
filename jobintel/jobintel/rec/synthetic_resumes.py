"""SYNTHETIC resumes with known ground truth, and a renderer that turns them into page images (clean or degraded).

Purpose: MEASURE the resume pipeline (NER fields, experience, OCR round trip) where no labelled real resumes exist.  Every resume is fictional (names, companies and
contacts come from small pools; e-mail domain example.com), generated from a seed, and flagged SYNTHETIC.  The numbers measured on them describe the template's difficulty,
not real-world accuracy: real resumes are messier (columns, icons, tables, creative layouts, other languages)."""
from __future__ import annotations

import datetime as dt
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LABEL = "SYNTHETIC resume - fictional, generated for pipeline measurement"
FIRST = ["Priya", "Rahul", "Anita", "Vikram", "Neha", "Arjun", "Meera", "Sanjay", "Kavya", "Rohan", "Isha", "Aman", "Divya", "Karan", "Pooja", "Nikhil"]
LAST = ["Sharma", "Verma", "Iyer", "Patel", "Nair", "Kulkarni", "Singh", "Reddy", "Joshi", "Mehta", "Gupta", "Desai", "Rao", "Menon", "Shah", "Pillai"]
COMPANIES = ["Northwind Analytics Pvt Ltd", "Contoso Technologies", "Fabrikam Systems", "Globex Solutions", "Initech Software", "Umbrella Services", "Hooli Labs", "Stark Consulting",
             "Wayne Group", "Acme Infotech", "Vandelay Private Limited", "Tyrell Corporation"]
SCHOOLS = ["Example University", "Demo College", "Test Institute", "Sample Institute of Technology", "Pune Institute of Computer Technology"]
TITLES = ["Data Analyst", "Software Engineer", "Business Analyst", "Backend Developer", "QA Engineer", "DevOps Engineer", "Junior Analyst", "Project Coordinator"]
DEGREES = ["B.Tech in Computer Science", "B.Sc. in Statistics", "MCA", "BCA", "B.E. in Information Technology"]
CERTS = ["AWS Cloud Practitioner", "Power BI Data Analyst Associate", "Certified Scrum Master", "Google Data Analytics Certificate", "Oracle Java SE Programmer"]
SKILL_POOL = ["python", "sql", "excel", "power bi", "tableau", "java", "spring boot", "docker", "kubernetes", "aws", "machine learning", "tensorflow", "react", "javascript",
              "git", "linux", "jenkins", "html", "css", "mongodb", "postgresql", "pandas", "scikit-learn"]
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class TruthResume:
    name: str
    email: str
    phone: str
    url: str
    city: str
    skills: List[str]
    companies: List[str]
    schools: List[str]
    certifications: List[str]
    experience_months: int
    stated_years: Optional[int]
    jobs: List[Dict] = field(default_factory=list)
    label: str = LABEL


def _fmt(y: int, m: int, style: int, end: bool = False) -> str:
    return [f"{MON[m - 1]} {y}", f"{m:02d}/{y}", f"{y}"][style]


def make_resume(rng: np.random.Generator, today: dt.date, skill_pool: Optional[Sequence[str]] = None) -> Tuple[str, TruthResume]:
    pool = list(skill_pool or SKILL_POOL)
    first, last = str(rng.choice(FIRST)), str(rng.choice(LAST))
    name = f"{first} {last}"
    email = f"{first.lower()}.{last.lower()}@example.com"
    phone = f"+91 {int(rng.integers(70000, 99999))} {int(rng.integers(10000, 99999))}"
    url = f"linkedin.com/in/{first.lower()}{last.lower()}"
    city = str(rng.choice(["Pune", "Mumbai", "Bengaluru", "Hyderabad"]))
    skills = [str(s) for s in rng.choice(pool, size=int(rng.integers(5, 10)), replace=False)]
    n_jobs = int(rng.integers(1, 4))
    style = int(rng.integers(0, 3))
    now_idx = today.year * 12 + today.month - 1                                     # month index of 'today' (year * 12 + month0)
    current = bool(rng.random() < 0.6)
    end_idx = now_idx if current else now_idx - int(rng.integers(1, 4))             # a current job runs through today's month; otherwise it ended 1-3 months ago
    jobs, intervals = [], []
    cur_end = end_idx
    for j in range(n_jobs):
        dur = int(rng.integers(8, 40))
        start = cur_end - dur + 1                                                   # both end months are INCLUSIVE ('Jan 2019 - Mar 2022' = 39 months)
        jobs.append({"title": str(rng.choice(TITLES)), "company": str(rng.choice(COMPANIES)), "start": start, "end": cur_end, "present": current and j == 0, "months": dur})
        intervals.append((start, cur_end))
        cur_end = start - int(rng.integers(2, 7))                                   # a gap of 1-5 months between jobs
    exp_months = sum(e - s + 1 for s, e in intervals)
    schools = [str(rng.choice(SCHOOLS))]
    certs = [str(c) for c in rng.choice(CERTS, size=int(rng.integers(0, 3)), replace=False)]
    stated = int(round(exp_months / 12)) if rng.random() < 0.5 and exp_months >= 12 else None
    lines = [name, f"{email} | {phone} | {url} | {city}, India", ""]
    lines += ["SUMMARY", (f"Professional with {stated} years of experience in " if stated else "Professional skilled in ") + ", ".join(skills[:3]) + ".", ""]
    lines += ["EXPERIENCE"]
    for j in jobs:
        sy, sm = divmod(j["start"], 12)
        ey, em = divmod(j["end"], 12)
        rng_txt = f"{_fmt(sy, sm + 1, style)} - " + ("Present" if j["present"] else _fmt(ey, em + 1, style, True))
        lines += [f"{j['title']}, {j['company']}   {rng_txt}", f"- Worked with {skills[int(rng.integers(len(skills)))]} on team projects"]
    lines += ["", "EDUCATION", f"{rng.choice(DEGREES)}, {schools[0]}   {int(today.year - 8)} - {int(today.year - 5)}", "", "SKILLS", ", ".join(skills)]
    if certs:
        lines += ["", "CERTIFICATIONS"] + [f"- {c}" for c in certs]
    truth = TruthResume(name, email, phone, url, city, skills, sorted({j["company"] for j in jobs}), schools, certs, exp_months, stated, jobs)
    return "\n".join(lines) + "\n", truth


# ------------------------------------------------------------------------------------------------------------------- rendering
def _fonts(size: int):
    from PIL import ImageFont
    base = "/usr/share/fonts/truetype/dejavu/"
    return ImageFont.truetype(base + "DejaVuSans.ttf", size), ImageFont.truetype(base + "DejaVuSans-Bold.ttf", size)


def render_resume_image(text: str, *, width: int = 1654, height: int = 2339, font_size: int = 26, columns: int = 1, rotate_deg: float = 0.0, noise_sigma: float = 0.0,
                        blur: float = 0.0, scale: float = 1.0, jpeg_quality: Optional[int] = None, seed: int = 0) -> Tuple[np.ndarray, str]:
    """page image (uint8 gray) + the ground-truth text in READING ORDER (left column first when columns = 2).  Degradations are applied in this order: rotate, blur, noise, downscale, jpeg."""
    import cv2
    from PIL import Image, ImageDraw
    reg, bold = _fonts(font_size)
    lines = text.rstrip("\n").split("\n")
    margin, lh = 110, int(font_size * 1.55)
    if columns == 2:
        wrap = 46
        cols = [[], []]
        half = max(1, len(lines) // 2)
        cut = next((i for i in range(half, len(lines)) if lines[i] == ""), half)
        for ci, chunk in enumerate((lines[:cut], lines[cut:])):
            for ln in chunk:
                cols[ci] += textwrap.wrap(ln, wrap) or [""]
        xs = [margin, width // 2 + 30]
    else:
        cols, xs = [[w for ln in lines for w in (textwrap.wrap(ln, 95) or [""])]], [margin]
    img = Image.new("L", (width, height), 255)
    d = ImageDraw.Draw(img)
    for ci, col in enumerate(cols):
        y = margin
        for ln in col:
            heading = ln.isupper() and len(ln) > 3 and " " not in ln.strip()
            d.text((xs[ci], y), ln, fill=0, font=bold if heading else reg)
            y += lh
    truth = "\n".join(ln for col in cols for ln in col)
    a = np.array(img)
    rng = np.random.default_rng(seed)
    if rotate_deg:
        M = cv2.getRotationMatrix2D((width / 2, height / 2), rotate_deg, 1.0)
        a = cv2.warpAffine(a, M, (width, height), flags=cv2.INTER_CUBIC, borderValue=255)
    if blur:
        a = cv2.GaussianBlur(a, (0, 0), blur)
    if noise_sigma:
        a = np.clip(a.astype(np.float32) + rng.normal(0, noise_sigma, a.shape), 0, 255).astype(np.uint8)
    if scale != 1.0:
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if jpeg_quality:
        a = cv2.imdecode(cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])[1], cv2.IMREAD_GRAYSCALE)
    return a, truth
