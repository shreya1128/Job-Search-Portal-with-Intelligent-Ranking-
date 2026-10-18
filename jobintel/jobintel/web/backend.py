"""Service layer of the web application: every heavy object is built ONCE (lazily, thread-safe) and shared by all requests.

Nothing here trains a model and nothing is rebuilt per search: the inverted index, the knowledge graph, the neural encoders and the collaborative model are loaded from the files
and tables the earlier phases produced.  A resource that cannot be loaded (missing file, no database, no TensorFlow) is recorded with the reason; the page that needs it says so
instead of inventing a result, and the pages that do not need it keep working.

    backend.index / .engine / .catalog / .kg / .analyzer / .dl / .encoder / .autoencoder / .cf / .market     -> the object, or raises Unavailable(reason)
    backend.try_get("dl")                                                                                      -> (object or None, reason or None)
    backend.status()                                                                                           -> {resource: {"state": "ready" | "not loaded" | "unavailable", ...}}
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .settings import WebSettings

log = logging.getLogger("jobintel.web")

RESOURCES = ("db", "index", "engine", "catalog", "kg", "analyzer", "dl", "encoder", "autoencoder", "cf", "market")
CF_TTL_SECONDS = 600


class Unavailable(RuntimeError):
    """A resource the page needs is not available; the message is shown to the user as the reason."""


@dataclass
class _Res:
    value: Any = None
    error: Optional[str] = None
    seconds: float = 0.0
    info: Optional[Dict] = None
    loaded_at: float = 0.0


class Backend:
    def __init__(self, settings: WebSettings, **injected: Any):
        self.settings = settings
        self._state: Dict[str, _Res] = {}
        self._locks: Dict[str, threading.RLock] = {n: threading.RLock() for n in RESOURCES}
        self.tf_lock = threading.Lock()                      # Keras predict() is not guaranteed thread-safe: one network call at a time
        self._injected_names = set(injected)
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_lock = threading.Lock()
        for name, value in injected.items():
            if name not in RESOURCES:
                raise TypeError(f"unknown resource {name!r}")
            self._state[name] = _Res(value=value, loaded_at=time.time())

    # ------------------------------------------------------------------------------------------------ generic machinery
    def _res(self, name: str, loader: Callable[[], Any]) -> _Res:
        r = self._state.get(name)
        if r is not None:
            return r
        with self._locks[name]:
            r = self._state.get(name)
            if r is None:
                t0 = time.perf_counter()
                try:
                    out = loader()
                    info = None
                    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict) and name == "cf":
                        out, info = out
                    r = _Res(value=out, seconds=round(time.perf_counter() - t0, 2), info=info, loaded_at=time.time())
                except Unavailable as e:
                    r = _Res(error=str(e), seconds=round(time.perf_counter() - t0, 2))
                except Exception as e:                       # noqa: BLE001 - the reason is shown to the user, the traceback goes to the log
                    log.exception("loading %s failed", name)
                    r = _Res(error=f"{type(e).__name__}: {e}", seconds=round(time.perf_counter() - t0, 2))
                self._state[name] = r
        return r

    def _need(self, name: str, loader: Callable[[], Any]) -> Any:
        r = self._res(name, loader)
        if r.error is not None:
            raise Unavailable(r.error)
        return r.value

    def try_get(self, name: str) -> Tuple[Any, Optional[str]]:
        try:
            return getattr(self, name), None
        except Unavailable as e:
            return None, str(e)

    def reset(self, name: str) -> None:
        with self._locks[name]:
            self._state.pop(name, None)

    # ------------------------------------------------------------------------------------------------ resources
    @property
    def db(self):
        def load():
            from sqlalchemy import text
            from ..db.session import get_engine
            eng = get_engine()
            with eng.connect() as c:
                c.execute(text("SELECT 1"))
            return eng
        return self._need("db", load)

    @property
    def index(self):
        def load():
            from ..ir.index import InvertedIndex
            p = self.settings.index_path
            if not Path(p).exists():
                raise Unavailable(f"the search index {p} does not exist; build it with  python scripts/build_index.py --write-db")
            return InvertedIndex.load(Path(p))
        return self._need("index", load)

    @property
    def engine(self):
        def load():
            from ..ir.engine import DbDocStore, SearchEngine
            store = DbDocStore(self.db)
            if self.settings.parts > 1:
                from ..ir.distributed import DistributedEngine
                return DistributedEngine.build(self.index, store, self.settings.parts, "threads")
            return SearchEngine(self.index, store)
        return self._need("engine", load)

    @property
    def catalog(self):
        def load():
            from ..rec.catalog import ItemCatalog
            return ItemCatalog(self.index)
        return self._need("catalog", load)

    @property
    def kg(self):
        def load():
            from ..rec.knowledge import KnowledgeGraph
            return KnowledgeGraph.from_db(self.db)
        return self._need("kg", load)

    @property
    def analyzer(self):
        def load():
            from ..rec.resume import ResumeAnalyzer
            return ResumeAnalyzer(self.engine.extractor)
        return self._need("analyzer", load)

    @property
    def dl(self):
        """DL skill-similarity signal (Phase 6 → Phase 8).

        Uses DLSkillSimilarity.from_files (the v2 denoising autoencoder) which maps a
        profile's skill multi-hot vector through the trained DAE bottleneck and computes
        cosine similarity against per-job latent vectors.  Falls back to DLSimilarity
        (Phase 5 text-space stub) only when the v2 artefacts are absent.  If neither is
        available, raises Unavailable so the hybrid recommender drops the signal and
        explains why — no placeholder score is ever produced.
        """
        def load():
            import os
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            from ..rec.dlsim import DLSkillSimilarity, DLSimilarity

            # --- preferred path: v2 skill-space DAE (3,887/3,951 docs covered in current index)
            d = DLSkillSimilarity.from_files(self.settings.data_dir, self.catalog, version="v2")
            if d.available:
                log.info("DL signal: DLSkillSimilarity v2 loaded (%d docs covered)", d.n_docs_covered)
                return d

            # --- fallback path: v1 (no version suffix)
            d1 = DLSkillSimilarity.from_files(self.settings.data_dir, self.catalog, version="")
            if d1.available:
                log.info("DL signal: DLSkillSimilarity v1 loaded (%d docs covered)", d1.n_docs_covered)
                return d1

            # --- last resort: Phase-5 text-space stub (legacy, normally off)
            d_txt = DLSimilarity.load_default(self.settings.data_dir)
            if d_txt.available:
                log.info("DL signal: DLSimilarity (Phase 5 text space) loaded")
                return d_txt

            # Nothing worked — surface the most recent reason
            raise Unavailable(d.reason or d1.reason or "DL artefacts not found; run scripts/train_autoencoder_v2.py")

        return self._need("dl", load)

    @property
    def encoder(self):
        """DLEncoder: best sequence model by validation loss (job-function prediction, text embeddings)."""
        def load():
            from ..dl.encoder import DLEncoder
            return DLEncoder.load(self.settings.data_dir / "dl")
        return self._need("encoder", load)

    @property
    def autoencoder(self):
        """(denoising autoencoder, vocabulary) for skill suggestions."""
        def load():
            d = self.settings.data_dir / "dl"
            f_model, f_vocab = d / "models" / "ae_denoising.keras", d / "ae_vocab.json"
            if not f_model.exists() or not f_vocab.exists():
                raise Unavailable("the skill autoencoder is not trained yet (run scripts/train_autoencoder.py)")
            import os
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import keras
            return keras.saving.load_model(f_model, compile=False), json.loads(f_vocab.read_text(encoding="utf-8"))["vocab"]
        return self._need("autoencoder", load)

    @property
    def cf(self):
        """Item-based collaborative filter fitted on the interactions table (re-fitted at most every CF_TTL_SECONDS); what it was fitted on is in ``cf_info``."""
        r = self._state.get("cf")
        if r is not None and r.error is None and r.loaded_at and time.time() - r.loaded_at > CF_TTL_SECONDS and not self._injected("cf"):
            self.reset("cf")

        def load():
            from sqlalchemy import select
            from ..db import models as M
            from ..rec.interactions import EVENT_WEIGHTS, ItemKNN
            ix = self.index
            with self.db.connect() as c:
                rows = c.execute(select(M.JobInteraction.user_id, M.JobInteraction.job_id, M.JobInteraction.viewed, M.JobInteraction.clicked, M.JobInteraction.saved,
                                        M.JobInteraction.applied, M.JobInteraction.is_synthetic)).all()
            users, items, w, synth = {}, [], [], 0
            u_idx: List[int] = []
            for uid, jid, v, c_, s, a, is_syn in rows:
                di = ix.doc_index(int(jid))
                if di is None:
                    continue
                wt = EVENT_WEIGHTS["apply"] if a else EVENT_WEIGHTS["save"] if s else EVENT_WEIGHTS["click"] if c_ else EVENT_WEIGHTS["view"] if v else 0.0
                if wt <= 0:
                    continue
                u_idx.append(users.setdefault(int(uid), len(users)))
                items.append(di)
                w.append(wt)
                synth += int(bool(is_syn))
            if len(w) < 50 or len(users) < 5:
                raise Unavailable(f"only {len(w)} usable interactions from {len(users)} users: too few to fit a collaborative model")
            knn = ItemKNN(ix.n_docs).fit(u_idx, items, w)
            info = {"interactions": len(w), "users": len(users), "synthetic_interactions": synth, "real_interactions": len(w) - synth,
                    "label": ("SYNTHETIC demo interactions" if synth == len(w) else "real and synthetic interactions mixed" if synth else "real interactions")}
            return knn, info
        return self._need("cf", load)

    def _injected(self, name: str) -> bool:
        return name in self._injected_names

    @property
    def cf_info(self) -> Optional[Dict]:
        r = self._state.get("cf")
        return r.info if r is not None else None

    @property
    def market(self) -> Dict:
        """Job-market aggregates (the MapReduce output in data/analytics/market.json).  Never computed implicitly: see ``compute_market`` / scripts/run_mapreduce.py."""
        def load():
            f = self.market_file
            if not f.exists():
                raise Unavailable("the market analytics have not been computed yet; run  python scripts/run_mapreduce.py  or use the button on this page")
            return json.loads(f.read_text(encoding="utf-8"))
        return self._need("market", load)

    @property
    def market_file(self) -> Path:
        return self.settings.data_dir / "analytics" / "market.json"

    def compute_market(self) -> Dict:
        """Run the market MapReduce job (single worker, about a minute for 40k jobs) and store the result; the next ``market`` access reads it."""
        from ..mapreduce import jobs as J
        from ..mapreduce.framework import run_mapreduce, sum_reducer
        from ..mapreduce.market import load_market_records
        recs = load_market_records(self.db)
        res = run_mapreduce(recs, J.market_mapper, sum_reducer, n_workers=1, chunk_size=2000, combiner=J.sum_combiner)
        out = J.assemble_market(res.output)
        f = self.market_file
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out, indent=1, default=list), encoding="utf-8")
        self.reset("market")
        return {"jobs": len(recs), "stats": res.stats}

    def facets(self) -> Dict:
        """Choices for the search filters, derived from the loaded index (computed once): top cities, countries, employment types, sources, most common skills."""
        with self._locks["catalog"]:
            f = getattr(self, "_facets", None)
            if f is None:
                ix = self.index
                top = lambda arr, n: [(str(v), int(c)) for v, c in sorted(zip(*np.unique(np.asarray(arr, dtype=object).astype(str), return_counts=True)), key=lambda x: -x[1])[:n] if str(v)]
                cities = sorted(((c, len(d)) for c, d in ix.city_docs.items()), key=lambda x: -x[1])[:80]
                skills = sorted(((s_, len(d)) for s_, d in ix.skill_docs.items()), key=lambda x: -x[1])[:400]
                f = {"cities": [(c.title(), n) for c, n in cities], "countries": top(ix.country_lc, 15), "employment": top(ix.employment_lc, 12),
                     "sources": top(ix.source_arr, 10), "skills": [s_ for s_, _ in skills], "areas": top(ix.cluster_arr, 20)}
                self._facets = f
            return f

    # ------------------------------------------------------------------------------------------------ home-page facts (cached; never load the index)
    def _cached(self, key: str, ttl: float, make: Callable[[], Any]) -> Any:
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None and time.time() - hit[0] < ttl:
                return hit[1]
        val = make()
        with self._cache_lock:
            self._cache[key] = (time.time(), val)
        return val

    def area_counts(self) -> List[Tuple[str, int]]:
        """[(job area, number of postings)] biggest first.  From the loaded index when it is ready (so the number matches what a search for that area returns), otherwise from the
        database with the same exclusions as the index (repeated and flagged-as-fake postings left out)."""
        r = self._state.get("index")
        if r is not None and r.error is None:
            return list(self.facets()["areas"])

        def make():
            from sqlalchemy import func, select
            from sqlalchemy.orm import Session
            from ..db import models as M
            with Session(self.db) as s:
                rows = s.execute(select(M.Job.cluster, func.count()).where(M.Job.duplicate_of.is_(None), M.Job.is_fraudulent.is_(False)).group_by(M.Job.cluster)).all()
            return sorted(((str(c), int(n)) for c, n in rows if c), key=lambda x: -x[1])
        return self._cached("area_counts", 600, make)

    _SENIORITY = re.compile(r"^(senior|sr\.?|lead|junior|jr\.?|principal|associate|staff|chief|head of)\s+", re.I)
    _PREFIX = re.compile(r"^(graduates?|urgent|hiring|immediate|wanted|entry level|fresher|freshers)\s*[:\-]\s*", re.I)
    _GENERIC = {"technical lead", "lead", "manager", "consultant", "specialist", "engineer", "analyst", "developer", "executive", "associate"}

    @classmethod
    def role_family(cls, title: str) -> str:
        """'Senior Data Engineer (Azure)' -> 'data engineer': the title without level words and bracketed or colon-separated detail.  A generic base such as 'Technical Lead (Data
        Engineer)' takes the bracketed words instead."""
        t = cls._PREFIX.sub("", (title or "").strip())
        paren = re.search(r"\(([^)]{3,40})\)", t)
        base = re.sub(r"\([^)]*\)", " ", t)
        base = re.split(r"\s[:|/-]\s|:", base)[0]
        base = re.sub(r"\s+", " ", base).strip().lower()
        for _ in range(2):
            base = cls._SENIORITY.sub("", base)
        base = base.strip(" -,")
        if paren and base in cls._GENERIC:
            base = re.sub(r"\s+", " ", paren.group(1)).strip().lower()
        return base

    def popular_roles(self, n: int = 8) -> List[Tuple[str, int]]:
        """The role names that most postings carry, level words removed, from the database.  [(role, postings)]"""
        def make():
            from collections import Counter
            from sqlalchemy import func, select
            from sqlalchemy.orm import Session
            from ..db import models as M
            with Session(self.db) as s:
                rows = s.execute(select(func.lower(M.Job.title), func.count()).where(M.Job.duplicate_of.is_(None), M.Job.is_fraudulent.is_(False))
                                 .group_by(func.lower(M.Job.title)).order_by(func.count().desc()).limit(1500)).all()
            c: Counter = Counter()
            for title, k in rows:
                fam = self.role_family(str(title))
                if 3 <= len(fam) <= 40:
                    c[fam] += int(k)
            return c.most_common(40)
        return self._cached("popular_roles", 3600, make)[:n]

    def example_jobs(self, per_area: int = 1, areas: int = 6) -> List[Dict]:
        """A few real postings for the home page: one from each of the biggest job areas, with a company name and a city.  The pick changes once a day and is
        not ordered by quality or date (the data has no reliable posting dates), which the page says."""
        import datetime as dt

        def make():
            from sqlalchemy import select
            from sqlalchemy.orm import Session
            from ..db import models as M
            seed = dt.date.today().toordinal()
            out: List[Dict] = []
            counts = [a for a, _ in self.area_counts() if a != "other"][:areas]
            with Session(self.db) as s:
                for i, area in enumerate(counts):
                    q = (select(M.Job.job_id, M.Job.title, M.Job.city, M.Job.country, M.Job.experience_text, M.Company.name)
                         .join(M.Company, M.Company.company_id == M.Job.company_id)
                         .where(M.Job.cluster == area, M.Job.duplicate_of.is_(None), M.Job.is_fraudulent.is_(False),
                                M.Company.name_is_derived.is_(False), M.Job.city.is_not(None)).order_by(M.Job.job_id).limit(4000))
                    rows = s.execute(q).all()
                    if not rows:
                        continue
                    for k in range(per_area):
                        jid, title, city, country, exp, company = rows[(seed * 7919 + i * 104729 + k * 31) % len(rows)]
                        skills = s.execute(select(M.Skill.canonical_name).join(M.JobSkill, M.JobSkill.skill_id == M.Skill.skill_id).where(M.JobSkill.job_id == jid)
                                           .order_by(M.JobSkill.mentions.desc(), M.Skill.canonical_name).limit(5)).scalars().all()
                        out.append({"job_id": int(jid), "title": title, "company": company, "city": city, "country": country, "experience": exp, "area": area, "skills": list(skills)})
            return out
        return self._cached("example_jobs", 3600, make)

    def index_summary(self) -> Optional[Dict]:
        """Facts about the loaded index for page footers (None while the index is not loaded: this never triggers loading)."""
        r = self._state.get("index")
        if r is None or r.error is not None:
            return None
        if r.info is None:
            ix = r.value
            r.info = {**ix.stats(), "sources": sorted({str(x) for x in ix.meta["source"]})}
        return r.info

    # ------------------------------------------------------------------------------------------------ status
    @staticmethod
    def tesseract_available() -> bool:
        return shutil.which("tesseract") is not None

    def status(self) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for n in RESOURCES:
            r = self._state.get(n)
            if r is None:
                out[n] = {"state": "not loaded"}
            elif r.error:
                out[n] = {"state": "unavailable", "reason": r.error}
            else:
                out[n] = {"state": "ready", "seconds": r.seconds, **({"info": r.info} if r.info else {})}
        out["ocr"] = {"state": "ready" if self.tesseract_available() else "unavailable", **({} if self.tesseract_available() else {"reason": "the tesseract program is not installed"})}
        return out

    def warm_up(self, names: Tuple[str, ...] = ("index", "engine", "catalog", "kg", "analyzer", "dl", "encoder", "autoencoder")) -> None:
        """Load resources in the calling thread (the app runs this in a background thread); failures are recorded, never raised.  The home-page facts are computed here too, so the
        first visitor does not wait for them."""
        for fn in (self.area_counts, self.popular_roles, self.example_jobs):
            try:
                fn()
            except Exception:                                # noqa: BLE001
                log.warning("warm-up of %s failed", fn.__name__, exc_info=True)
        for n in names:
            self.try_get(n)

    def close(self) -> None:
        r = self._state.get("engine")
        if r is not None and r.value is not None and hasattr(r.value, "close"):
            try:
                r.value.close()
            except Exception:                                # noqa: BLE001
                log.exception("closing the engine failed")
