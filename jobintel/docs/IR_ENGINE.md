# IR engine (Phase 2) - algorithms, formulas, measured behaviour

```
jobs (MySQL) --scripts/build_index.py--> InvertedIndex (file, separate from the DB)      jobs+links --scripts/compute_pagerank.py--> job_pagerank
USER QUERY -> QueryProcessor (validation DFA + entity automata) -> filters -> CANDIDATES (docs with >= 1 query term)
           -> baseline | LM | bigram LM | DFR            rank all candidates with ONE model
           -> hybrid: shortlist = top-100 LM U top-100 DFR -> + passage + skill + location + experience + PageRank -> weighted score
           -> explained top-k (matched terms, skills, location, best passage with <mark> highlighting, every score)
```
Measured on the real collection: **60,808 documents, 84,106 terms, 6.9 M tokens, 4.7 M postings** (naukri 11.9k, emscad 16.2k, mcf 18.7k, linkedin 14.0k; duplicates, fraudulent and synthetic jobs excluded). Reproduce with `scripts/verify_phase2.py` -> `docs/phase2_report.md`.

| component | objective | processing | math | measured / limits |
|---|---|---|---|---|
| **Index** (`ir/index.py`) | the searchable collection: document lookup, term lookup, postings, df, cf, passage lookup | tokens = Phase-1 pipeline (lower-case, tech-aware, stop-words, Porter); postings in CSR arrays sorted by doc; token sequences kept for bigrams | df(t) = #docs containing t; cf(t) = total occurrences; P(w\|C) = cf/\|C\| | builds in 12 s, 109 MB file. Rebuild after re-ingesting. |
| **Query processing** (`ir/query.py`) | turn text into a query representation and reject bad input | DFA over character classes (L letter/digit, S space, P allowed punctuation, X other -> dead state) validates; automata find skills/city/level/experience/employment; those entity words are removed from the text terms; generic words (jobs, hiring ...) dropped | - | `Entry level machine learning jobs in Pune` -> terms {machin, learn}, skill, location Pune, experience 0-2 (level). A query with no text terms ("jobs in Pune") uses the city as a restriction. |
| **Keyword baseline** | simple reference ranking | rank by number of distinct query terms present, ties by total term frequency | - | the baseline for the Phase-4 comparison |
| **Unigram language model** (`models.unigram_lm`) | rank by how likely the document's language model generates the query | multinomial per document; Dirichlet (default, mu = 200) or Jelinek-Mercer (lambda = 0.3) smoothing | log P(q\|d) = sum_w qtf(w) log P(w\|d);  Dirichlet P(w\|d) = (tf + mu P(w\|C)) / (\|d\| + mu);  JM P(w\|d) = (1-lambda) tf/\|d\| + lambda P(w\|C) | a word missing from a document keeps non-zero probability (collection back-off); a query word absent from the whole collection carries no evidence and is skipped (reported as a warning). mu = 200 was chosen for short documents (mean length 114), **not tuned**. |
| **Bigram language model** | account for word order ("machine learning" vs "learning machine") | P(w1\|d) unigram; P(w_i\|w_{i-1},d) = (c(w_{i-1}w_i;d) + mu_b P_C(w_i\|w_{i-1})) / (c(w_{i-1};d) + mu_b); P_C backs off to the unigram collection model (alpha = 0.5) | as left | separates two documents with identical words in a different order, which the unigram model cannot (unit test). 16 ms median. Trigrams are not implemented. |
| **DFR** (`models.dfr`) | how much a term's frequency in a document diverges from chance | normalisation 2: tfn = tf log2(1 + c avg_len/\|d\|); **PL2** (Poisson randomness, Laplace after-effect) or **InL2** (inverse document frequency) | PL2: w = (tfn log2(tfn/lambda) + (lambda - tfn) log2 e + 0.5 log2(2 pi tfn)) / (tfn + 1), lambda = cf/N, clipped at 0; InL2: w = tfn/(tfn+1) log2((N+1)/(df+0.5)) | matches hand computation in tests. Clipping at 0 is our choice (a term not more frequent than chance gives no evidence). c = 1 not tuned. |
| **Passage retrieval** (`ir/passages.py`) | show WHY a job matched, not just that it did | split into passages (30-90 words) along section headings, bullets and sentences; score = mean over query terms of log P(w\|passage) - log P(w\|C); best passage highlighted | passage LM with Dirichlet mu = 50 | folds fragments < 8 words into neighbours; 84k passages stored in `job_passages`. Skills-only sources have one passage (the skills line). MyCareersFuture text is lower-case without stop-words, so its highlighted passage looks that way. |
| **Hybrid ranking** (`ir/ranker.py`) | combine textual relevance with other evidence | every signal min-max normalised to [0,1]; signals that do not apply (no city typed ...) are dropped and the weights re-normalised | score = sum w_i s_i / sum active w_i; defaults LM .25, DFR .15, passage .12, skill .12, location .20, experience .08, PageRank .08 | defaults are **provisional and not claimed optimal**; they are configurable per query and Phase 4 evaluates alternatives. Scores are relative to the candidate pool (a 3-candidate query makes the best look like 1.00). |
| **PageRank** (`ir/pagerank.py`) | an authority signal from link structure | power iteration, damping 0.85, dangling mass spread uniformly; graph Company -> Career page -> Job -> Related job (+ Job -> Company) | PR(v) = (1-d)/n + d (sum_{u->v} PR(u) w/W(u) + dangling/n) | 94k nodes, 214k edges, converged in 127 iterations; verified against a direct linear-system solution. Edges for the datasets are CONSTRUCTED (company<->jobs from the data, related jobs = same cluster + shared rare skills; `job_links.is_constructed = 1`); crawler edges are observed. **Correlation of PageRank with the employer's number of postings is 0.65**: here it measures prominence/volume (big employers, recruiters, job boards), not quality. Its usefulness for retrieval is an experiment for Phase 4. |

## Measured on the real index (16 queries x 3 runs)
| model | median ms | p95 ms |
|---|---|---|
| baseline / LM / DFR | 3.8 / 4.2 / 4.3 | 8.3 / 7.8 / 7.2 |
| bigram LM | 16.3 | 50.0 |
| passage / hybrid (fetch text of ~200 shortlisted jobs from MySQL) | 35.0 / 33.9 | 86.3 / 77.8 |

Mean overlap of the top-10 sets between models ranges from 0.09 (baseline vs DFR) to 0.60 (LM vs hybrid): the models rank differently. This is **not** a statement about which is better.

## Known limitations
* Relevance is unmeasured until Phase 4 (needs hand-labelled judgments).
* Mixed collection: skills-only documents (Naukri, LinkedIn without summaries) are ~15-50 tokens, full-text ones ~170; length normalisation helps but cross-source ranking is imperfect. Use the `sources` / `text_quality` filters when comparing.
* Stemming is Porter (English); no synonyms ("ML" is expanded only through the skill lexicon).
* LinkedIn summaries are not loaded yet, so its passages are skill lists.
* Single process; parallel/cluster retrieval is Phase 3.
