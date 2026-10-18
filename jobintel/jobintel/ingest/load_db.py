"""Ingestion pipeline:  JobRecord stream  ->  clean / tokenise / extract / de-duplicate  ->  MySQL.

For every record we keep BOTH representations required by the spec:
    description_original  exactly what the source supplied (may contain HTML)
    description_clean     display text (no HTML, repaired unicode, paragraphs kept)  -> passage highlighting
    processed_text        title + body, lower-cased, stop-words removed, Porter-stemmed -> indexing
Skill tags supplied by the dataset and skills found by the automata in the text are merged into job_skills
(origin = dataset_tag | extracted | both).  Duplicates are FLAGGED (duplicate_of), never silently dropped.
"""
from __future__ import annotations

import datetime as dt
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.engine import Engine

from jobintel.db.models import Company, CrawlerSource, Job, JobSkill, Skill
from jobintel.extraction.extractor import Extractor
from jobintel.extraction.lexicon import SkillLexicon, load_lexicon
from jobintel.ingest import common as C
from jobintel.ingest.loaders import SOURCE_META
from jobintel.processing.dedupe import NearDuplicateIndex, content_hash, normalize_for_hash, simhash
from jobintel.processing.text import clean_description, index_tokens
from jobintel.records import JobRecord


# sources whose skill tags are too noisy to accept blindly
STRICT_TAG_SOURCES = {"linkedin"}


def _cut(v, n):
    return v[:n] if isinstance(v, str) else v


@dataclass
class _State:
    next_job_id: int
    skill_ids: Dict[str, int]
    company_ids: Dict[str, int]
    hash_to_job: Dict[str, int] = field(default_factory=dict)
    near: NearDuplicateIndex = field(default_factory=NearDuplicateIndex)
    seen_ext: set = field(default_factory=set)
    tkey: Dict[int, tuple] = field(default_factory=dict)
    stats: Counter = field(default_factory=Counter)


class Ingestor:
    def __init__(self, engine: Engine, extractor: Optional[Extractor] = None, lexicon: Optional[SkillLexicon] = None,
                 batch: int = 200, verbose: bool = True):
        self.engine = engine
        self.lex = lexicon or load_lexicon()
        self.extractor = extractor or Extractor(self.lex)
        self.batch = batch            # keep modest: LONGTEXT rows are large and MySQL's max_allowed_packet may be small
        self.verbose = verbose
        self._dialect = engine.dialect.name

    # ------------------------------------------------------------------ db helpers
    def _ins(self, table):
        st = insert(table)
        return st.prefix_with("OR IGNORE", dialect="sqlite").prefix_with("IGNORE", dialect="mysql")

    def _ensure_source(self, conn, name: str) -> int:
        meta = dict(SOURCE_META.get(name, dict(source_type="dataset", text_quality="full")))
        row = conn.execute(select(CrawlerSource.source_id).where(CrawlerSource.name == name)).first()
        if row:
            conn.execute(update(CrawlerSource).where(CrawlerSource.source_id == row[0]).values(**meta))
            return row[0]
        conn.execute(insert(CrawlerSource).values(name=name, **meta))
        return conn.execute(select(CrawlerSource.source_id).where(CrawlerSource.name == name)).scalar_one()

    def _ensure_companies(self, conn, pending: Dict[str, dict], st: _State) -> None:
        new = {k: v for k, v in pending.items() if k not in st.company_ids}
        if not new:
            return
        conn.execute(self._ins(Company), list(new.values()))
        keys = list(new)
        for i in range(0, len(keys), 400):
            for cid, nn in conn.execute(select(Company.company_id, Company.normalized_name).where(Company.normalized_name.in_(keys[i:i + 400]))):
                st.company_ids[nn.lower()] = cid
        for k in keys:                      # collation-equal keys resolve to the same row
            st.company_ids.setdefault(k, st.company_ids.get(k.lower()))

    def _ensure_skills(self, conn, pending: Dict[str, dict], st: _State) -> None:
        new = {k: v for k, v in pending.items() if k not in st.skill_ids}
        if not new:
            return
        conn.execute(self._ins(Skill), list(new.values()))
        keys = list(new)
        for i in range(0, len(keys), 400):
            for sid, cn in conn.execute(select(Skill.skill_id, Skill.canonical_name).where(Skill.canonical_name.in_(keys[i:i + 400]))):
                st.skill_ids[cn.lower()] = sid
        for k in keys:
            st.skill_ids.setdefault(k, st.skill_ids.get(k.lower()))

    # ------------------------------------------------------------------ per-record processing
    def _skills_for(self, rec: JobRecord, ex: Optional[dict]) -> Dict[str, dict]:
        """canonical -> {type, origin, mentions, ambiguous}"""
        out: Dict[str, dict] = {}
        kinds = rec.extras.get("tag_kinds", {})
        for tag in rec.skills:
            kind = kinds.get(tag, "skill")
            if kind in ("noise", "experience"):
                continue
            canon = self.lex.normalize(tag)
            if not canon:
                continue
            ent = self.lex.get(canon)
            if kind == "skill" and rec.source in STRICT_TAG_SOURCES and ent is None:
                continue          # LinkedIn tags are LLM-extracted and very noisy: keep only tags the lexicon vouches for
            typ = {"certification": "certification", "education": "education"}.get(kind) or (ent.type if ent else "other")
            out[canon] = dict(type=typ, origin="dataset_tag", mentions=0, ambiguous=bool(ent and ent.ambiguous))
        if ex:
            for s in ex["skills"]:
                cur = out.get(s["name"])
                if cur:
                    cur["origin"], cur["mentions"] = "both", s["mentions"]
                else:
                    ent = self.lex.get(s["name"])
                    out[s["name"]] = dict(type=s["type"], origin="extracted", mentions=s["mentions"], ambiguous=bool(ent and ent.ambiguous))
        return out

    def _process(self, rec: JobRecord, st: _State) -> Optional[dict]:
        if rec.company and C.is_placeholder_company(rec.company):
            rec.company, rec.company_profile = None, None            # 'Company Undisclosed' is not an employer
        raw = rec.description if rec.description and rec.description.strip() else None
        clean = clean_description(raw) if raw else ""
        ex = self.extractor.extract(clean) if clean and rec.text_quality in ("full", "preprocessed") else None
        skills = self._skills_for(rec, ex)

        if rec.text_quality == "skills_only" or not clean:
            display = "Skills: " + ", ".join(rec.skills) if rec.skills else ""
            body_for_index = " ".join(rec.skills)
            quality = "skills_only" if rec.skills else "none"
        else:
            display, body_for_index, quality = clean, clean, rec.text_quality
        tokens = index_tokens(rec.title + "\n" + body_for_index)

        # ---- experience: dataset value > extracted from text > seniority band --------------------------------
        e_min, e_max, e_txt, e_src = rec.experience_min, rec.experience_max, rec.experience_text, rec.experience_source
        if e_src != "dataset" and ex and ex["experience"]:
            e_min, e_max, e_txt, e_src = ex["experience"]["min"], ex["experience"]["max"], ex["experience"]["text"], "extracted"
        # ---- salary / education / employment type -------------------------------------------------------------
        s_min, s_max, s_cur, s_per = rec.salary_min, rec.salary_max, rec.salary_currency, rec.salary_period
        if s_min is None and ex and ex["salary"]:
            sv = ex["salary"]
            s_min, s_max, s_cur, s_per = sv["min"], sv["max"], (SOURCE_META.get(rec.source, {}).get("currency") or sv.get("currency")), sv.get("period")
        education = rec.education or (self.extractor.best_education(ex["education"]) if ex else None)
        emp = rec.employment_type or (ex["employment_type"] if ex else None)

        # ---- duplicates ---------------------------------------------------------------------------------------
        job_id = st.next_job_id
        st.next_job_id += 1
        comp = rec.company or ""
        # A duplicate = same title AND same employer AND same place AND (identical or near-identical text).  Identical boilerplate used for
        # different cities/titles is a distinct listing and must stay searchable.
        place = rec.city or rec.location_raw
        tkey = (normalize_for_hash(rec.title), normalize_for_hash(rec.company), normalize_for_hash(place))
        if quality in ("full", "preprocessed"):
            h = content_hash(rec.title, comp, place, clean[:3000])
        else:
            h = content_hash(rec.title, comp, rec.location_raw, rec.experience_text, " ".join(sorted(skills)))
        dup_of = st.hash_to_job.get(h)
        sim = None
        if len(tokens) >= 40 and quality != "skills_only":
            sim = simhash(tokens)
            if dup_of is None:
                dup_of = next((c for c in st.near.candidates(sim) if st.tkey.get(c) == tkey), None)
            if dup_of is None:
                st.near.add(job_id, sim)
        st.tkey[job_id] = tkey
        st.hash_to_job.setdefault(h, job_id)
        if dup_of is not None:
            st.stats["duplicates"] += 1

        ext, k = rec.external_id[:240], 1
        while ext in st.seen_ext:                      # guarantee UNIQUE(source_id, external_id) whatever the source gives us
            k += 1
            ext = f"{rec.external_id[:230]}#{k}"
        st.seen_ext.add(ext)

        job = dict(
            job_id=job_id, external_id=_cut(ext, 255), title=_cut(rec.title, 500), location_raw=_cut(rec.location_raw, 500),
            city=_cut(rec.city, 120), cities=rec.cities or None, region=_cut(rec.region, 120), country=_cut(rec.country, 80),
            description_original=raw, description_clean=display or None, processed_text=" ".join(tokens) or None, n_tokens=len(tokens),
            text_quality=quality, experience_min=e_min, experience_max=e_max, experience_text=_cut(e_txt, 160), experience_source=e_src,
            education=_cut(education, 160), salary_min=s_min, salary_max=s_max, salary_currency=_cut(s_cur, 8), salary_period=s_per,
            employment_type=_cut(emp, 60), seniority=_cut(rec.seniority, 80), category=_cut(rec.category, 200), industry=_cut(rec.industry, 200),
            cluster=C.cluster_for_title(rec.title), posting_date=rec.posting_date, url=_cut(rec.url, 1000), content_hash=h,
            simhash=f"{sim:016x}" if sim is not None else None, duplicate_of=dup_of, is_fraudulent=bool(rec.is_fraudulent),
            _company=rec.company, _profile=rec.company_profile, _derived=rec.company_name_is_derived, _skills=skills)
        for k, v in (("experience", e_src), ("salary", s_min), ("education", education)):
            st.stats[f"with_{k}"] += 1 if v else 0
        st.stats[f"quality_{quality}"] += 1
        st.stats["skills_dataset_tag"] += sum(1 for v in skills.values() if v["origin"] in ("dataset_tag", "both"))
        st.stats["skills_extracted"] += sum(1 for v in skills.values() if v["origin"] in ("extracted", "both"))
        return job

    # ------------------------------------------------------------------ flush one batch
    def _flush(self, conn, batch: List[dict], source_id: int, st: _State) -> None:
        pending_c: Dict[str, dict] = {}
        pending_s: Dict[str, dict] = {}
        for j in batch:
            if j["_company"]:
                nn = C.normalize_company(j["_company"])[:300]
                j["_cnorm"] = nn
                if nn not in st.company_ids and nn not in pending_c:
                    pending_c[nn] = dict(name=_cut(j["_company"], 300), normalized_name=nn, profile_text=_cut(j["_profile"], 20000), name_is_derived=bool(j["_derived"]))
                elif j["_profile"] and nn in pending_c and not pending_c[nn]["profile_text"]:
                    pending_c[nn]["profile_text"] = _cut(j["_profile"], 20000)
            for canon, v in j["_skills"].items():
                if canon not in st.skill_ids and canon not in pending_s:
                    pending_s[canon] = dict(name=_cut(canon, 200), canonical_name=_cut(canon, 200), skill_type=v["type"], source="dataset", ambiguous=v["ambiguous"])
        self._ensure_companies(conn, pending_c, st)
        self._ensure_skills(conn, pending_s, st)

        jobs_rows, js_rows = [], []
        for j in batch:
            skills = j.pop("_skills")
            cn = j.pop("_cnorm", None)
            j.pop("_company"), j.pop("_profile"), j.pop("_derived")
            j["source_id"] = source_id
            j["company_id"] = st.company_ids.get(cn) if cn else None
            jobs_rows.append(j)
            seen_sid = set()
            for canon, v in skills.items():
                sid = st.skill_ids.get(canon) or st.skill_ids.get(canon.lower())
                if sid and sid not in seen_sid:
                    seen_sid.add(sid)
                    js_rows.append(dict(job_id=j["job_id"], skill_id=sid, origin=v["origin"], mentions=v["mentions"]))
        # duplicate_of references an earlier job id that is already inserted (earlier batch) or in this batch (earlier row)
        conn.execute(insert(Job), jobs_rows)
        if js_rows:
            conn.execute(self._ins(JobSkill), js_rows)

    # ------------------------------------------------------------------ public API
    def ingest(self, source: str, records: Iterable[JobRecord], *, replace: bool = True, limit: Optional[int] = None) -> Dict:
        t0 = time.time()
        with self.engine.begin() as conn:
            source_id = self._ensure_source(conn, source)
            if replace:
                conn.execute(delete(Job).where(Job.source_id == source_id))
            st = _State(
                next_job_id=(conn.execute(select(func.coalesce(func.max(Job.job_id), 0))).scalar_one() + 1),
                skill_ids={n.lower(): i for i, n in conn.execute(select(Skill.skill_id, Skill.canonical_name))},
                company_ids={n.lower(): i for i, n in conn.execute(select(Company.company_id, Company.normalized_name))})
        n_in, batch = 0, []
        for rec in records:
            if limit and n_in >= limit:
                break
            n_in += 1
            job = self._process(rec, st)
            if job:
                batch.append(job)
            if len(batch) >= self.batch:
                with self.engine.begin() as conn:
                    self._flush(conn, batch, source_id, st)
                batch = []
                if self.verbose and n_in % 2000 < self.batch:
                    print(f"  [{source}] {n_in:,} records  ({time.time() - t0:.0f}s)", flush=True)
        if batch:
            with self.engine.begin() as conn:
                self._flush(conn, batch, source_id, st)
        with self.engine.begin() as conn:
            conn.execute(update(CrawlerSource).where(CrawlerSource.source_id == source_id).values(n_records=n_in, last_run_at=dt.datetime.now()))
        stats = dict(st.stats)
        stats.update(source=source, records=n_in, seconds=round(time.time() - t0, 1))
        return stats
