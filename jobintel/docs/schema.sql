-- Auto-generated from jobintel/db/models.py  (MySQL / MariaDB)
SET NAMES utf8mb4;

CREATE TABLE companies (
	company_id INTEGER NOT NULL AUTO_INCREMENT, 
	name VARCHAR(300) NOT NULL, 
	normalized_name VARCHAR(300) NOT NULL, 
	industry VARCHAR(200), 
	career_page_url VARCHAR(1000), 
	profile_text TEXT, 
	name_is_derived BOOL NOT NULL, 
	CONSTRAINT pk_companies PRIMARY KEY (company_id), 
	CONSTRAINT uq_companies_normalized_name UNIQUE (normalized_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE crawler_sources (
	source_id INTEGER NOT NULL AUTO_INCREMENT, 
	name VARCHAR(80) NOT NULL, 
	source_type ENUM('dataset','crawler','synthetic') NOT NULL, 
	base_url VARCHAR(500), 
	description TEXT, 
	license_note TEXT, 
	citation TEXT, 
	country VARCHAR(80), 
	currency VARCHAR(8), 
	robots_respected BOOL NOT NULL, 
	text_quality ENUM('full','skills_only','preprocessed','none') NOT NULL, 
	n_records INTEGER NOT NULL, 
	is_active BOOL NOT NULL, 
	last_run_at DATETIME, 
	config JSON, 
	CONSTRAINT pk_crawler_sources PRIMARY KEY (source_id), 
	CONSTRAINT uq_crawler_sources_name UNIQUE (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE ir_experiments (
	exp_id INTEGER NOT NULL AUTO_INCREMENT, 
	run_label VARCHAR(80) NOT NULL, 
	query_text VARCHAR(1000) NOT NULL, 
	ranking_model ENUM('keyword_baseline','language_model_unigram','language_model_bigram','dfr','passage','hybrid_ir','hybrid_ir_pagerank','hybrid_full') NOT NULL, 
	k INTEGER NOT NULL, 
	precision_at_k FLOAT, 
	recall_at_k FLOAT, 
	mrr FLOAT, 
	ndcg FLOAT, 
	latency_ms FLOAT, 
	n_docs INTEGER, 
	execution_mode ENUM('sequential','parallel'), 
	relevance_source VARCHAR(60), 
	config JSON, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	CONSTRAINT pk_ir_experiments PRIMARY KEY (exp_id), 
	CONSTRAINT ck_ir_experiments_k_pos CHECK (k > 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_ir_experiments_model ON ir_experiments (ranking_model);
CREATE INDEX ix_ir_experiments_run ON ir_experiments (run_label);

CREATE TABLE knowledge_relationships (
	rel_id BIGINT NOT NULL AUTO_INCREMENT, 
	subject_type ENUM('skill','occupation') NOT NULL, 
	subject_id INTEGER NOT NULL, 
	relation ENUM('requires_essential','requires_optional','uses_technology','related_to','broader_than','part_of','co_occurs_with','similar_to','typical_skill') NOT NULL, 
	object_type ENUM('skill','occupation') NOT NULL, 
	object_id INTEGER NOT NULL, 
	weight FLOAT NOT NULL, 
	source VARCHAR(30) NOT NULL, 
	CONSTRAINT pk_knowledge_relationships PRIMARY KEY (rel_id), 
	CONSTRAINT uq_kg_edge UNIQUE (subject_type, subject_id, relation, object_type, object_id, source), 
	CONSTRAINT ck_knowledge_relationships_weight_nonneg CHECK (weight >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_kg_object ON knowledge_relationships (object_type, object_id);
CREATE INDEX ix_kg_subject ON knowledge_relationships (subject_type, subject_id);

CREATE TABLE model_experiments (
	exp_id INTEGER NOT NULL AUTO_INCREMENT, 
	model_type ENUM('rnn','birnn','lstm','gru','autoencoder','dae','deep_ae','cae','vae','baseline') NOT NULL, 
	task VARCHAR(120) NOT NULL, 
	dataset VARCHAR(120), 
	hyperparameters JSON, 
	accuracy FLOAT, 
	precision_macro FLOAT, 
	recall_macro FLOAT, 
	f1_macro FLOAT, 
	val_loss FLOAT, 
	reconstruction_loss FLOAT, 
	train_seconds FLOAT, 
	n_parameters INTEGER, 
	artifact_path VARCHAR(500), 
	notes TEXT, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	CONSTRAINT pk_model_experiments PRIMARY KEY (exp_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_model_experiments_type ON model_experiments (model_type);

CREATE TABLE occupations (
	occupation_id INTEGER NOT NULL AUTO_INCREMENT, 
	name VARCHAR(300) NOT NULL, 
	source ENUM('esco','onet','curated') NOT NULL, 
	external_id VARCHAR(300) NOT NULL, 
	description TEXT, 
	job_zone INTEGER, 
	aliases JSON, 
	CONSTRAINT pk_occupations PRIMARY KEY (occupation_id), 
	CONSTRAINT uq_occupations_source_ext UNIQUE (source, external_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_occupations_name ON occupations (name(120));

CREATE TABLE skills (
	skill_id INTEGER NOT NULL AUTO_INCREMENT, 
	name VARCHAR(200) NOT NULL, 
	canonical_name VARCHAR(200) NOT NULL, 
	skill_type ENUM('technical','tool','soft','knowledge','certification','education','other') NOT NULL, 
	aliases JSON, 
	source VARCHAR(40) NOT NULL, 
	esco_uri VARCHAR(300), 
	ambiguous BOOL NOT NULL, 
	CONSTRAINT pk_skills PRIMARY KEY (skill_id), 
	CONSTRAINT uq_skills_canonical_name UNIQUE (canonical_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_skills_esco_uri ON skills (esco_uri);
CREATE INDEX ix_skills_type ON skills (skill_type);

CREATE TABLE users (
	user_id INTEGER NOT NULL AUTO_INCREMENT, 
	username VARCHAR(80) NOT NULL, 
	email VARCHAR(255) NOT NULL, 
	password_hash VARCHAR(255) NOT NULL, 
	`role` ENUM('user','admin') NOT NULL, 
	is_synthetic BOOL NOT NULL, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	CONSTRAINT pk_users PRIMARY KEY (user_id), 
	CONSTRAINT uq_users_username UNIQUE (username), 
	CONSTRAINT uq_users_email UNIQUE (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE jobs (
	job_id BIGINT NOT NULL AUTO_INCREMENT, 
	source_id INTEGER NOT NULL, 
	external_id VARCHAR(255) NOT NULL, 
	title VARCHAR(500) NOT NULL, 
	company_id INTEGER, 
	location_raw VARCHAR(500), 
	city VARCHAR(120), 
	cities JSON, 
	region VARCHAR(120), 
	country VARCHAR(80), 
	description_original LONGTEXT, 
	description_clean LONGTEXT, 
	processed_text LONGTEXT, 
	n_tokens INTEGER NOT NULL, 
	text_quality ENUM('full','skills_only','preprocessed','none') NOT NULL, 
	experience_min FLOAT, 
	experience_max FLOAT, 
	experience_text VARCHAR(160), 
	experience_source ENUM('dataset','extracted','seniority_map'), 
	education VARCHAR(160), 
	salary_min FLOAT, 
	salary_max FLOAT, 
	salary_currency VARCHAR(8), 
	salary_period ENUM('hour','day','week','month','year'), 
	employment_type VARCHAR(60), 
	seniority VARCHAR(80), 
	category VARCHAR(200), 
	industry VARCHAR(200), 
	cluster VARCHAR(30), 
	posting_date DATE, 
	url VARCHAR(1000), 
	content_hash VARCHAR(40), 
	simhash VARCHAR(16), 
	duplicate_of BIGINT, 
	is_fraudulent BOOL NOT NULL, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	CONSTRAINT pk_jobs PRIMARY KEY (job_id), 
	CONSTRAINT uq_jobs_source_external UNIQUE (source_id, external_id), 
	CONSTRAINT ck_jobs_exp_min_nonneg CHECK (experience_min IS NULL OR experience_min >= 0), 
	CONSTRAINT ck_jobs_exp_order CHECK (experience_max IS NULL OR experience_min IS NULL OR experience_max >= experience_min), 
	CONSTRAINT ck_jobs_salary_order CHECK (salary_max IS NULL OR salary_min IS NULL OR salary_max >= salary_min), 
	CONSTRAINT fk_jobs_source_id_crawler_sources FOREIGN KEY(source_id) REFERENCES crawler_sources (source_id) ON DELETE RESTRICT, 
	CONSTRAINT fk_jobs_company_id_companies FOREIGN KEY(company_id) REFERENCES companies (company_id) ON DELETE SET NULL, 
	CONSTRAINT fk_jobs_duplicate_of_jobs FOREIGN KEY(duplicate_of) REFERENCES jobs (job_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_jobs_category ON jobs (category);
CREATE INDEX ix_jobs_city ON jobs (city);
CREATE INDEX ix_jobs_cluster ON jobs (cluster);
CREATE INDEX ix_jobs_company ON jobs (company_id);
CREATE INDEX ix_jobs_content_hash ON jobs (content_hash);
CREATE INDEX ix_jobs_duplicate_of ON jobs (duplicate_of);
CREATE INDEX ix_jobs_posting_date ON jobs (posting_date);
CREATE INDEX ix_jobs_title ON jobs (title(120));

CREATE TABLE searches (
	search_id BIGINT NOT NULL AUTO_INCREMENT, 
	user_id INTEGER, 
	query_text VARCHAR(1000) NOT NULL, 
	parsed_query JSON, 
	filters JSON, 
	ranking_model VARCHAR(40), 
	n_results INTEGER NOT NULL, 
	latency_ms FLOAT, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	CONSTRAINT pk_searches PRIMARY KEY (search_id), 
	CONSTRAINT fk_searches_user_id_users FOREIGN KEY(user_id) REFERENCES users (user_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_searches_created ON searches (created_at);
CREATE INDEX ix_searches_user ON searches (user_id);

CREATE TABLE user_profiles (
	profile_id INTEGER NOT NULL AUTO_INCREMENT, 
	user_id INTEGER NOT NULL, 
	full_name VARCHAR(200), 
	experience_years FLOAT, 
	exp_range_min FLOAT, 
	exp_range_max FLOAT, 
	preferred_role VARCHAR(200), 
	preferred_location VARCHAR(120), 
	preferred_categories JSON, 
	education VARCHAR(120), 
	resume_filename VARCHAR(300), 
	resume_text LONGTEXT, 
	summary TEXT, 
	updated_at DATETIME NOT NULL DEFAULT (now()), 
	CONSTRAINT pk_user_profiles PRIMARY KEY (profile_id), 
	CONSTRAINT ck_user_profiles_exp_nonneg CHECK (experience_years IS NULL OR experience_years >= 0), 
	CONSTRAINT uq_user_profiles_user_id UNIQUE (user_id), 
	CONSTRAINT fk_user_profiles_user_id_users FOREIGN KEY(user_id) REFERENCES users (user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE user_skills (
	user_id INTEGER NOT NULL, 
	skill_id INTEGER NOT NULL, 
	source ENUM('manual','resume') NOT NULL, 
	confidence FLOAT NOT NULL, 
	CONSTRAINT pk_user_skills PRIMARY KEY (user_id, skill_id), 
	CONSTRAINT fk_user_skills_user_id_users FOREIGN KEY(user_id) REFERENCES users (user_id) ON DELETE CASCADE, 
	CONSTRAINT fk_user_skills_skill_id_skills FOREIGN KEY(skill_id) REFERENCES skills (skill_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE job_interactions (
	interaction_id BIGINT NOT NULL AUTO_INCREMENT, 
	user_id INTEGER NOT NULL, 
	job_id BIGINT NOT NULL, 
	viewed BOOL NOT NULL, 
	clicked BOOL NOT NULL, 
	saved BOOL NOT NULL, 
	applied BOOL NOT NULL, 
	is_synthetic BOOL NOT NULL, 
	last_event_at DATETIME NOT NULL DEFAULT (now()), 
	CONSTRAINT pk_job_interactions PRIMARY KEY (interaction_id), 
	CONSTRAINT uq_job_interactions_user_job UNIQUE (user_id, job_id), 
	CONSTRAINT fk_job_interactions_user_id_users FOREIGN KEY(user_id) REFERENCES users (user_id) ON DELETE CASCADE, 
	CONSTRAINT fk_job_interactions_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (job_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_job_interactions_job ON job_interactions (job_id);

CREATE TABLE job_links (
	link_id BIGINT NOT NULL AUTO_INCREMENT, 
	link_type ENUM('company_to_job','job_to_company','related_job') NOT NULL, 
	src_company_id INTEGER, 
	src_job_id BIGINT, 
	dst_company_id INTEGER, 
	dst_job_id BIGINT, 
	is_constructed BOOL NOT NULL, 
	weight FLOAT NOT NULL, 
	CONSTRAINT pk_job_links PRIMARY KEY (link_id), 
	CONSTRAINT ck_job_links_weight_nonneg CHECK (weight >= 0), 
	CONSTRAINT ck_job_links_endpoints_match_type CHECK ((link_type = 'company_to_job' AND src_company_id IS NOT NULL AND dst_job_id IS NOT NULL) OR (link_type = 'job_to_company' AND src_job_id IS NOT NULL AND dst_company_id IS NOT NULL) OR (link_type = 'related_job' AND src_job_id IS NOT NULL AND dst_job_id IS NOT NULL)), 
	CONSTRAINT fk_job_links_src_company_id_companies FOREIGN KEY(src_company_id) REFERENCES companies (company_id) ON DELETE CASCADE, 
	CONSTRAINT fk_job_links_src_job_id_jobs FOREIGN KEY(src_job_id) REFERENCES jobs (job_id) ON DELETE CASCADE, 
	CONSTRAINT fk_job_links_dst_company_id_companies FOREIGN KEY(dst_company_id) REFERENCES companies (company_id) ON DELETE CASCADE, 
	CONSTRAINT fk_job_links_dst_job_id_jobs FOREIGN KEY(dst_job_id) REFERENCES jobs (job_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_job_links_dst_job ON job_links (dst_job_id);
CREATE INDEX ix_job_links_src_company ON job_links (src_company_id);
CREATE INDEX ix_job_links_src_job ON job_links (src_job_id);

CREATE TABLE job_pagerank (
	job_id BIGINT NOT NULL, 
	pagerank FLOAT NOT NULL, 
	company_pagerank FLOAT, 
	graph_version VARCHAR(40) NOT NULL, 
	computed_at DATETIME NOT NULL DEFAULT (now()), 
	CONSTRAINT pk_job_pagerank PRIMARY KEY (job_id), 
	CONSTRAINT ck_job_pagerank_pr_nonneg CHECK (pagerank >= 0), 
	CONSTRAINT fk_job_pagerank_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (job_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE job_passages (
	passage_id BIGINT NOT NULL AUTO_INCREMENT, 
	job_id BIGINT NOT NULL, 
	passage_no INTEGER NOT NULL, 
	passage_text TEXT NOT NULL, 
	n_tokens INTEGER NOT NULL, 
	section VARCHAR(80), 
	CONSTRAINT pk_job_passages PRIMARY KEY (passage_id), 
	CONSTRAINT uq_job_passages_job_no UNIQUE (job_id, passage_no), 
	CONSTRAINT fk_job_passages_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (job_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE job_skills (
	job_id BIGINT NOT NULL, 
	skill_id INTEGER NOT NULL, 
	origin ENUM('dataset_tag','extracted','both') NOT NULL, 
	mentions INTEGER NOT NULL, 
	CONSTRAINT pk_job_skills PRIMARY KEY (job_id, skill_id), 
	CONSTRAINT fk_job_skills_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (job_id) ON DELETE CASCADE, 
	CONSTRAINT fk_job_skills_skill_id_skills FOREIGN KEY(skill_id) REFERENCES skills (skill_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_job_skills_skill ON job_skills (skill_id);

CREATE TABLE job_terms (
	job_id BIGINT NOT NULL, 
	term VARCHAR(100) NOT NULL, 
	tf INTEGER NOT NULL, 
	CONSTRAINT pk_job_terms PRIMARY KEY (job_id, term), 
	CONSTRAINT ck_job_terms_tf_pos CHECK (tf > 0), 
	CONSTRAINT fk_job_terms_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (job_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_job_terms_term ON job_terms (term);

CREATE TABLE recommendations (
	rec_id BIGINT NOT NULL AUTO_INCREMENT, 
	user_id INTEGER NOT NULL, 
	job_id BIGINT NOT NULL, 
	position INTEGER NOT NULL, 
	ir_score FLOAT, 
	dl_score FLOAT, 
	content_score FLOAT, 
	collaborative_score FLOAT, 
	knowledge_score FLOAT, 
	skill_match_score FLOAT, 
	preference_score FLOAT, 
	pagerank_score FLOAT, 
	final_score FLOAT NOT NULL, 
	weights JSON, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	CONSTRAINT pk_recommendations PRIMARY KEY (rec_id), 
	CONSTRAINT fk_recommendations_user_id_users FOREIGN KEY(user_id) REFERENCES users (user_id) ON DELETE CASCADE, 
	CONSTRAINT fk_recommendations_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (job_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_recommendations_job ON recommendations (job_id);
CREATE INDEX ix_recommendations_user ON recommendations (user_id, created_at);

CREATE TABLE skill_gaps (
	gap_id BIGINT NOT NULL AUTO_INCREMENT, 
	user_id INTEGER NOT NULL, 
	job_id BIGINT NOT NULL, 
	skill_id INTEGER NOT NULL, 
	status ENUM('matching','missing') NOT NULL, 
	importance FLOAT, 
	market_frequency FLOAT, 
	created_at DATETIME NOT NULL DEFAULT (now()), 
	CONSTRAINT pk_skill_gaps PRIMARY KEY (gap_id), 
	CONSTRAINT uq_skill_gaps_u_j_s UNIQUE (user_id, job_id, skill_id), 
	CONSTRAINT fk_skill_gaps_user_id_users FOREIGN KEY(user_id) REFERENCES users (user_id) ON DELETE CASCADE, 
	CONSTRAINT fk_skill_gaps_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (job_id) ON DELETE CASCADE, 
	CONSTRAINT fk_skill_gaps_skill_id_skills FOREIGN KEY(skill_id) REFERENCES skills (skill_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_skill_gaps_job ON skill_gaps (job_id);

CREATE TABLE search_results (
	result_id BIGINT NOT NULL AUTO_INCREMENT, 
	search_id BIGINT NOT NULL, 
	job_id BIGINT NOT NULL, 
	position INTEGER NOT NULL, 
	lm_score FLOAT, 
	dfr_score FLOAT, 
	passage_score FLOAT, 
	pagerank_score FLOAT, 
	skill_score FLOAT, 
	location_score FLOAT, 
	experience_score FLOAT, 
	final_score FLOAT NOT NULL, 
	best_passage_id BIGINT, 
	explanation JSON, 
	CONSTRAINT pk_search_results PRIMARY KEY (result_id), 
	CONSTRAINT uq_search_results_pos UNIQUE (search_id, position), 
	CONSTRAINT fk_search_results_search_id_searches FOREIGN KEY(search_id) REFERENCES searches (search_id) ON DELETE CASCADE, 
	CONSTRAINT fk_search_results_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (job_id) ON DELETE CASCADE, 
	CONSTRAINT fk_search_results_best_passage_id_job_passages FOREIGN KEY(best_passage_id) REFERENCES job_passages (passage_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
CREATE INDEX ix_search_results_job ON search_results (job_id);
