"""Relational schema (single source of truth). Works on MySQL/MariaDB (target) and SQLite (tests only).

21 tables = the 20 required by the spec + `occupations` (roles needed by the knowledge base).
Design notes
  * jobs keeps BOTH the original description (description_original) and the processed
    representation (description_clean for display, processed_text for indexing).
  * duplicates are flagged (duplicate_of) rather than deleted, so analytics/index can exclude them
    while the audit trail is preserved.
  * job_links holds only edges whose two ends are real rows (company->job, job->job, job->company);
    the intermediate "career page" node of the PageRank graph is derived per company in code.
  * knowledge_relationships is polymorphic (skill|occupation on both ends); MySQL cannot enforce a
    polymorphic FK, so the loader validates ids and the verify script checks integrity.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import (JSON, BigInteger, Boolean, CheckConstraint, Date, DateTime, Enum, Float,
                        ForeignKey, Index, Integer, MetaData, String, Text, UniqueConstraint, func)
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)


LongText = Text().with_variant(mysql.LONGTEXT(), "mysql", "mariadb")
BigPK = BigInteger().with_variant(Integer(), "sqlite")  # SQLite only autoincrements INTEGER PKs

SKILL_TYPES = ("technical", "tool", "soft", "knowledge", "certification", "education", "other")
TEXT_QUALITY = ("full", "skills_only", "preprocessed", "none")
EXP_SOURCES = ("dataset", "extracted", "seniority_map")
PERIODS = ("hour", "day", "week", "month", "year")
KG_NODE = ("skill", "occupation")
KG_REL = ("requires_essential", "requires_optional", "uses_technology", "related_to", "broader_than",
          "part_of", "co_occurs_with", "similar_to", "typical_skill")
MODEL_TYPES = ("rnn", "birnn", "lstm", "gru", "autoencoder", "dae", "deep_ae", "cae", "vae", "baseline")
IR_MODELS = ("keyword_baseline", "language_model_unigram", "language_model_bigram", "dfr", "passage",
             "hybrid_ir", "hybrid_ir_pagerank", "hybrid_full")


# --------------------------------------------------------------------------- users
class User(Base):
    __tablename__ = "users"
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(Enum("user", "admin", name="user_role"), default="user", nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False,
                                               doc="True for generated demo users (synthetic interactions)")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    profile: Mapped[Optional["UserProfile"]] = relationship(back_populates="user", uselist=False,
                                                            cascade="all, delete-orphan", passive_deletes=True)


class UserProfile(Base):
    __tablename__ = "user_profiles"
    __table_args__ = (CheckConstraint("experience_years IS NULL OR experience_years >= 0", name="exp_nonneg"),)
    profile_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), unique=True, nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(200))
    experience_years: Mapped[Optional[float]] = mapped_column(Float)
    exp_range_min: Mapped[Optional[float]] = mapped_column(Float)
    exp_range_max: Mapped[Optional[float]] = mapped_column(Float)
    preferred_role: Mapped[Optional[str]] = mapped_column(String(200))
    preferred_location: Mapped[Optional[str]] = mapped_column(String(120))
    preferred_categories: Mapped[Optional[list]] = mapped_column(JSON)
    education: Mapped[Optional[str]] = mapped_column(String(120))
    resume_filename: Mapped[Optional[str]] = mapped_column(String(300))
    resume_text: Mapped[Optional[str]] = mapped_column(LongText)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    user: Mapped[User] = relationship(back_populates="profile")


# --------------------------------------------------------------------------- knowledge
class Skill(Base):
    __tablename__ = "skills"
    skill_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    canonical_name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    skill_type: Mapped[str] = mapped_column(Enum(*SKILL_TYPES, name="skill_type"), default="technical", nullable=False)
    aliases: Mapped[Optional[list]] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String(40), nullable=False, default="dataset")
    esco_uri: Mapped[Optional[str]] = mapped_column(String(300))
    ambiguous: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False,
                                            doc="Generic single word: kept for tag normalisation but not used for free-text extraction")
    __table_args__ = (Index("ix_skills_type", "skill_type"), Index("ix_skills_esco_uri", "esco_uri"))


class Occupation(Base):
    __tablename__ = "occupations"
    occupation_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    source: Mapped[str] = mapped_column(Enum("esco", "onet", "curated", name="occupation_source"), nullable=False)
    external_id: Mapped[str] = mapped_column(String(300), nullable=False, doc="ESCO concept URI / O*NET-SOC code / curated slug")
    description: Mapped[Optional[str]] = mapped_column(Text)
    job_zone: Mapped[Optional[int]] = mapped_column(Integer, doc="O*NET job zone 1-5 (preparation level)")
    aliases: Mapped[Optional[list]] = mapped_column(JSON)
    __table_args__ = (UniqueConstraint("source", "external_id", name="uq_occupations_source_ext"),
                      Index("ix_occupations_name", "name", mysql_length=120))


class KnowledgeRelationship(Base):
    __tablename__ = "knowledge_relationships"
    rel_id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    subject_type: Mapped[str] = mapped_column(Enum(*KG_NODE, name="kg_subject_type"), nullable=False)
    subject_id: Mapped[int] = mapped_column(Integer, nullable=False)
    relation: Mapped[str] = mapped_column(Enum(*KG_REL, name="kg_relation"), nullable=False)
    object_type: Mapped[str] = mapped_column(Enum(*KG_NODE, name="kg_object_type"), nullable=False)
    object_id: Mapped[int] = mapped_column(Integer, nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    source: Mapped[str] = mapped_column(String(30), nullable=False)
    __table_args__ = (
        UniqueConstraint("subject_type", "subject_id", "relation", "object_type", "object_id", "source", name="uq_kg_edge"),
        Index("ix_kg_subject", "subject_type", "subject_id"),
        Index("ix_kg_object", "object_type", "object_id"),
        CheckConstraint("weight >= 0", name="weight_nonneg"),
    )


# --------------------------------------------------------------------------- sources & companies
class CrawlerSource(Base):
    __tablename__ = "crawler_sources"
    source_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    source_type: Mapped[str] = mapped_column(Enum("dataset", "crawler", "synthetic", name="source_type"), nullable=False)
    base_url: Mapped[Optional[str]] = mapped_column(String(500))
    description: Mapped[Optional[str]] = mapped_column(Text)
    license_note: Mapped[Optional[str]] = mapped_column(Text)
    citation: Mapped[Optional[str]] = mapped_column(Text)
    country: Mapped[Optional[str]] = mapped_column(String(80))
    currency: Mapped[Optional[str]] = mapped_column(String(8))
    robots_respected: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    text_quality: Mapped[str] = mapped_column(Enum(*TEXT_QUALITY, name="source_text_quality"), default="full")
    n_records: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_run_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime)
    config: Mapped[Optional[dict]] = mapped_column(JSON)


class Company(Base):
    __tablename__ = "companies"
    company_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(300), unique=True, nullable=False)
    industry: Mapped[Optional[str]] = mapped_column(String(200))
    career_page_url: Mapped[Optional[str]] = mapped_column(String(1000))
    profile_text: Mapped[Optional[str]] = mapped_column(Text)
    name_is_derived: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False,
                                                  doc="True when the dataset has no company name (EMSCAD: derived from profile-text hash)")


# --------------------------------------------------------------------------- jobs
class Job(Base):
    __tablename__ = "jobs"
    job_id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("crawler_sources.source_id", ondelete="RESTRICT"), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    company_id: Mapped[Optional[int]] = mapped_column(ForeignKey("companies.company_id", ondelete="SET NULL"))
    location_raw: Mapped[Optional[str]] = mapped_column(String(500))
    city: Mapped[Optional[str]] = mapped_column(String(120))
    cities: Mapped[Optional[list]] = mapped_column(JSON, doc="all normalised cities when a posting lists several")
    region: Mapped[Optional[str]] = mapped_column(String(120))
    country: Mapped[Optional[str]] = mapped_column(String(80))
    description_original: Mapped[Optional[str]] = mapped_column(LongText)
    description_clean: Mapped[Optional[str]] = mapped_column(LongText)
    processed_text: Mapped[Optional[str]] = mapped_column(LongText)
    n_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    text_quality: Mapped[str] = mapped_column(Enum(*TEXT_QUALITY, name="job_text_quality"), default="full", nullable=False)
    experience_min: Mapped[Optional[float]] = mapped_column(Float)
    experience_max: Mapped[Optional[float]] = mapped_column(Float)
    experience_text: Mapped[Optional[str]] = mapped_column(String(160))
    experience_source: Mapped[Optional[str]] = mapped_column(Enum(*EXP_SOURCES, name="exp_source"))
    education: Mapped[Optional[str]] = mapped_column(String(160))
    salary_min: Mapped[Optional[float]] = mapped_column(Float)
    salary_max: Mapped[Optional[float]] = mapped_column(Float)
    salary_currency: Mapped[Optional[str]] = mapped_column(String(8))
    salary_period: Mapped[Optional[str]] = mapped_column(Enum(*PERIODS, name="salary_period"))
    employment_type: Mapped[Optional[str]] = mapped_column(String(60))
    seniority: Mapped[Optional[str]] = mapped_column(String(80))
    category: Mapped[Optional[str]] = mapped_column(String(200), doc="label supplied by the dataset (e.g. EMSCAD function)")
    industry: Mapped[Optional[str]] = mapped_column(String(200))
    cluster: Mapped[Optional[str]] = mapped_column(String(30), doc="title-keyword cluster (software_it/data_ai/finance/healthcare/other) - a HEURISTIC, not ground truth")
    posting_date: Mapped[Optional[dt.date]] = mapped_column(Date)
    url: Mapped[Optional[str]] = mapped_column(String(1000))
    content_hash: Mapped[Optional[str]] = mapped_column(String(40))
    simhash: Mapped[Optional[str]] = mapped_column(String(16))
    duplicate_of: Mapped[Optional[int]] = mapped_column(ForeignKey("jobs.job_id", ondelete="SET NULL"))
    is_fraudulent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("source_id", "external_id", name="uq_jobs_source_external"),
        CheckConstraint("experience_min IS NULL OR experience_min >= 0", name="exp_min_nonneg"),
        CheckConstraint("experience_max IS NULL OR experience_min IS NULL OR experience_max >= experience_min", name="exp_order"),
        CheckConstraint("salary_max IS NULL OR salary_min IS NULL OR salary_max >= salary_min", name="salary_order"),
        Index("ix_jobs_company", "company_id"),
        Index("ix_jobs_city", "city"),
        Index("ix_jobs_category", "category"),
        Index("ix_jobs_cluster", "cluster"),
        Index("ix_jobs_posting_date", "posting_date"),
        Index("ix_jobs_content_hash", "content_hash"),
        Index("ix_jobs_duplicate_of", "duplicate_of"),
        Index("ix_jobs_title", "title", mysql_length=120),
    )


class JobSkill(Base):
    __tablename__ = "job_skills"
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.job_id", ondelete="CASCADE"), primary_key=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.skill_id", ondelete="CASCADE"), primary_key=True)
    origin: Mapped[str] = mapped_column(Enum("dataset_tag", "extracted", "both", name="skill_origin"), nullable=False)
    mentions: Mapped[int] = mapped_column(Integer, default=0, nullable=False, doc="occurrences in the description text")
    __table_args__ = (Index("ix_job_skills_skill", "skill_id"),)


class JobPassage(Base):
    __tablename__ = "job_passages"
    passage_id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False)
    passage_no: Mapped[int] = mapped_column(Integer, nullable=False)
    passage_text: Mapped[str] = mapped_column(Text, nullable=False)
    n_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    section: Mapped[Optional[str]] = mapped_column(String(80))
    __table_args__ = (UniqueConstraint("job_id", "passage_no", name="uq_job_passages_job_no"),)


class JobTerm(Base):
    __tablename__ = "job_terms"
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.job_id", ondelete="CASCADE"), primary_key=True)
    term: Mapped[str] = mapped_column(String(100), primary_key=True)
    tf: Mapped[int] = mapped_column(Integer, nullable=False)
    __table_args__ = (Index("ix_job_terms_term", "term"), CheckConstraint("tf > 0", name="tf_pos"))


class JobLink(Base):
    __tablename__ = "job_links"
    link_id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    link_type: Mapped[str] = mapped_column(Enum("company_to_job", "job_to_company", "related_job", name="link_type"), nullable=False)
    src_company_id: Mapped[Optional[int]] = mapped_column(ForeignKey("companies.company_id", ondelete="CASCADE"))
    src_job_id: Mapped[Optional[int]] = mapped_column(ForeignKey("jobs.job_id", ondelete="CASCADE"))
    dst_company_id: Mapped[Optional[int]] = mapped_column(ForeignKey("companies.company_id", ondelete="CASCADE"))
    dst_job_id: Mapped[Optional[int]] = mapped_column(ForeignKey("jobs.job_id", ondelete="CASCADE"))
    is_constructed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False,
                                                 doc="True = built by us (e.g. shared skills); False = hyperlink observed by the crawler")
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    __table_args__ = (
        Index("ix_job_links_src_job", "src_job_id"),
        Index("ix_job_links_dst_job", "dst_job_id"),
        Index("ix_job_links_src_company", "src_company_id"),
        CheckConstraint("weight >= 0", name="weight_nonneg"),
        CheckConstraint(
            "(link_type = 'company_to_job' AND src_company_id IS NOT NULL AND dst_job_id IS NOT NULL) OR "
            "(link_type = 'job_to_company' AND src_job_id IS NOT NULL AND dst_company_id IS NOT NULL) OR "
            "(link_type = 'related_job' AND src_job_id IS NOT NULL AND dst_job_id IS NOT NULL)", name="endpoints_match_type"),
    )


class JobPageRank(Base):
    __tablename__ = "job_pagerank"
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.job_id", ondelete="CASCADE"), primary_key=True)
    pagerank: Mapped[float] = mapped_column(Float, nullable=False)
    company_pagerank: Mapped[Optional[float]] = mapped_column(Float)
    graph_version: Mapped[str] = mapped_column(String(40), nullable=False)
    computed_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    __table_args__ = (CheckConstraint("pagerank >= 0", name="pr_nonneg"),)


# --------------------------------------------------------------------------- user behaviour & outputs
class UserSkill(Base):
    __tablename__ = "user_skills"
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.skill_id", ondelete="CASCADE"), primary_key=True)
    source: Mapped[str] = mapped_column(Enum("manual", "resume", name="user_skill_source"), default="manual", nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)


class JobInteraction(Base):
    __tablename__ = "job_interactions"
    interaction_id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False)
    viewed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    clicked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    saved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    applied: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False,
                                               doc="True = generated demo data, NOT real user behaviour")
    last_event_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    __table_args__ = (UniqueConstraint("user_id", "job_id", name="uq_job_interactions_user_job"),
                      Index("ix_job_interactions_job", "job_id"))


class Search(Base):
    __tablename__ = "searches"
    search_id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.user_id", ondelete="SET NULL"))
    query_text: Mapped[str] = mapped_column(String(1000), nullable=False)
    parsed_query: Mapped[Optional[dict]] = mapped_column(JSON)
    filters: Mapped[Optional[dict]] = mapped_column(JSON)
    ranking_model: Mapped[Optional[str]] = mapped_column(String(40))
    n_results: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    __table_args__ = (Index("ix_searches_user", "user_id"), Index("ix_searches_created", "created_at"))


class SearchResult(Base):
    __tablename__ = "search_results"
    result_id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    search_id: Mapped[int] = mapped_column(ForeignKey("searches.search_id", ondelete="CASCADE"), nullable=False)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    lm_score: Mapped[Optional[float]] = mapped_column(Float)
    dfr_score: Mapped[Optional[float]] = mapped_column(Float)
    passage_score: Mapped[Optional[float]] = mapped_column(Float)
    pagerank_score: Mapped[Optional[float]] = mapped_column(Float)
    skill_score: Mapped[Optional[float]] = mapped_column(Float)
    location_score: Mapped[Optional[float]] = mapped_column(Float)
    experience_score: Mapped[Optional[float]] = mapped_column(Float)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    best_passage_id: Mapped[Optional[int]] = mapped_column(ForeignKey("job_passages.passage_id", ondelete="SET NULL"))
    explanation: Mapped[Optional[dict]] = mapped_column(JSON)
    __table_args__ = (UniqueConstraint("search_id", "position", name="uq_search_results_pos"),
                      Index("ix_search_results_job", "job_id"))


class Recommendation(Base):
    __tablename__ = "recommendations"
    rec_id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    ir_score: Mapped[Optional[float]] = mapped_column(Float)
    dl_score: Mapped[Optional[float]] = mapped_column(Float)
    content_score: Mapped[Optional[float]] = mapped_column(Float)
    collaborative_score: Mapped[Optional[float]] = mapped_column(Float)
    knowledge_score: Mapped[Optional[float]] = mapped_column(Float)
    skill_match_score: Mapped[Optional[float]] = mapped_column(Float)
    preference_score: Mapped[Optional[float]] = mapped_column(Float)
    pagerank_score: Mapped[Optional[float]] = mapped_column(Float)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    weights: Mapped[Optional[dict]] = mapped_column(JSON, doc="the (configurable, NOT claimed optimal) weights used")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    __table_args__ = (Index("ix_recommendations_user", "user_id", "created_at"), Index("ix_recommendations_job", "job_id"))


class SkillGap(Base):
    __tablename__ = "skill_gaps"
    gap_id: Mapped[int] = mapped_column(BigPK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.job_id", ondelete="CASCADE"), nullable=False)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.skill_id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(Enum("matching", "missing", name="gap_status"), nullable=False)
    importance: Mapped[Optional[float]] = mapped_column(Float)
    market_frequency: Mapped[Optional[float]] = mapped_column(Float, doc="share of similar jobs that list this skill")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    __table_args__ = (UniqueConstraint("user_id", "job_id", "skill_id", name="uq_skill_gaps_u_j_s"),
                      Index("ix_skill_gaps_job", "job_id"))


# --------------------------------------------------------------------------- experiments
class ModelExperiment(Base):
    __tablename__ = "model_experiments"
    exp_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(Enum(*MODEL_TYPES, name="dl_model_type"), nullable=False)
    task: Mapped[str] = mapped_column(String(120), nullable=False)
    dataset: Mapped[Optional[str]] = mapped_column(String(120))
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON)
    accuracy: Mapped[Optional[float]] = mapped_column(Float)
    precision_macro: Mapped[Optional[float]] = mapped_column(Float)
    recall_macro: Mapped[Optional[float]] = mapped_column(Float)
    f1_macro: Mapped[Optional[float]] = mapped_column(Float)
    val_loss: Mapped[Optional[float]] = mapped_column(Float)
    reconstruction_loss: Mapped[Optional[float]] = mapped_column(Float)
    train_seconds: Mapped[Optional[float]] = mapped_column(Float)
    n_parameters: Mapped[Optional[int]] = mapped_column(Integer)
    artifact_path: Mapped[Optional[str]] = mapped_column(String(500))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    __table_args__ = (Index("ix_model_experiments_type", "model_type"),)


class IRExperiment(Base):
    __tablename__ = "ir_experiments"
    exp_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_label: Mapped[str] = mapped_column(String(80), nullable=False)
    query_text: Mapped[str] = mapped_column(String(1000), nullable=False)
    ranking_model: Mapped[str] = mapped_column(String(60), nullable=False, doc="evaluated system, e.g. keyword_baseline, lm_dirichlet, dfr_pl2, hybrid, hybrid_no_pagerank")
    k: Mapped[int] = mapped_column(Integer, nullable=False)
    precision_at_k: Mapped[Optional[float]] = mapped_column(Float)
    recall_at_k: Mapped[Optional[float]] = mapped_column(Float)
    mrr: Mapped[Optional[float]] = mapped_column(Float)
    ndcg: Mapped[Optional[float]] = mapped_column(Float)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float)
    n_docs: Mapped[Optional[int]] = mapped_column(Integer)
    execution_mode: Mapped[Optional[str]] = mapped_column(Enum("sequential", "parallel", name="exec_mode"))
    relevance_source: Mapped[Optional[str]] = mapped_column(String(60), doc="manual | proxy_skill_labels (must be reported separately)")
    config: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    __table_args__ = (Index("ix_ir_experiments_run", "run_label"), Index("ix_ir_experiments_model", "ranking_model"),
                      CheckConstraint("k > 0", name="k_pos"))


ALL_TABLES = [t.name for t in Base.metadata.sorted_tables]
