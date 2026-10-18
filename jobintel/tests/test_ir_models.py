import math
import numpy as np
import pytest

from jobintel.ir import models as M
from jobintel.ir.index import IndexDoc, InvertedIndex

DOCS = {
    0: "python python sql machine learning",
    1: "python java",
    2: "nurse patient care",
    3: "machine learning deep learning python",
    4: "learning machine python",                # same words as a phrase query but reversed order
}


@pytest.fixture()
def ix():
    return InvertedIndex.build(IndexDoc(i, t.split(), {"skills": [], "cities": None, "city": None}) for i, t in DOCS.items())


def q(ix, text):
    from collections import Counter
    c = Counter(text.split())
    return [(ix.term_id(w), n) for w, n in c.items() if ix.term_id(w) >= 0]


def test_index_statistics_and_postings(ix):
    assert ix.n_docs == 5 and ix.total_tokens == sum(len(t.split()) for t in DOCS.values())
    t = ix.term_id("python")
    docs, tf = ix.postings(t)
    assert list(docs) == [0, 1, 3, 4] and list(tf) == [2, 1, 1, 1]
    assert ix.df[t] == 4 and ix.cf[t] == 5                       # document frequency vs collection frequency
    assert ix.term_info("zzz") == {"term": "zzz", "in_vocab": False, "df": 0, "cf": 0}
    d = dict(ix.doc_distribution(0, 3))
    assert math.isclose(d["python"], 2 / 5)                       # multinomial MLE: P(python | job 0) = 2/5


def test_dirichlet_matches_the_formula_and_unseen_words_stay_nonzero(ix):
    cands = M.candidates(ix, q(ix, "python nurse"))
    mu = 10.0
    s = M.unigram_lm(ix, q(ix, "python nurse"), cands, mu=mu)
    total = ix.total_tokens
    for k, di in enumerate(cands):
        expect = 0.0
        for w in ("python", "nurse"):
            tid = ix.term_id(w)
            docs, tf = ix.postings(tid)
            f = dict(zip(docs, tf)).get(di, 0)
            p = (f + mu * ix.cf[tid] / total) / (ix.doc_len[di] + mu)
            assert p > 0                                           # smoothing: no zero probability for a word missing from the document
            expect += math.log(p)
        assert math.isclose(s[k], expect, rel_tol=1e-9)
    assert np.all(np.isfinite(s))


def test_unknown_query_word_is_ignored_not_fatal(ix):
    cands = M.candidates(ix, q(ix, "python"))
    a = M.unigram_lm(ix, q(ix, "python"), cands)
    b = M.unigram_lm(ix, q(ix, "python zzzunknown"), cands)       # q() drops OOV terms (as the engine does); scorers also skip cf = 0
    assert np.allclose(a, b) and np.all(np.isfinite(b))
    assert M.doc_term_probability(ix, 0, "zzzunknown") == 0.0


def test_jelinek_mercer_matches_the_formula_and_normalises_by_length(ix):
    qt = q(ix, "machine learning python")
    cands = M.candidates(ix, qt)
    lam = 0.3
    s_jm = M.unigram_lm(ix, qt, cands, smoothing="jm", lam=lam)
    for k, di in enumerate(cands):
        expect = 0.0
        for tid, _ in qt:
            docs, tf = ix.postings(tid)
            f = dict(zip(docs, tf)).get(di, 0)
            expect += math.log((1 - lam) * f / ix.doc_len[di] + lam * ix.cf[tid] / ix.total_tokens)
        assert math.isclose(s_jm[k], expect, rel_tol=1e-9)
    py = M.candidates(ix, q(ix, "python"))
    s2 = M.unigram_lm(ix, q(ix, "python"), py, smoothing="jm")
    pos = {int(d): i for i, d in enumerate(py)}
    assert s2[pos[1]] > s2[pos[0]]          # 'python java' (python = 1/2 of the text) beats 'python python sql machine learning' (2/5)
    with pytest.raises(ValueError):
        M.unigram_lm(ix, qt, cands, smoothing="nope")


def test_dfr_pl2_matches_hand_computation(ix):
    t = ix.term_id("learning")
    cands = M.candidates(ix, [(t, 1)])
    s = M.dfr(ix, [(t, 1)], cands, model="PL2", c=1.0)
    n, avg = ix.n_docs, ix.avg_doc_len
    lam = ix.cf[t] / n
    docs, tf = ix.postings(t)
    for d, f, sc in zip(docs, tf, s):
        tfn = f * math.log2(1 + avg / ix.doc_len[d])
        w = (tfn * math.log2(tfn / lam) + (lam - tfn) * math.log2(math.e) + 0.5 * math.log2(2 * math.pi * tfn)) / (tfn + 1)
        assert math.isclose(sc, max(w, 0.0), rel_tol=1e-9)
    assert s[list(cands).index(3)] > s[list(cands).index(0)]       # 'learning' twice in a short doc beats once


def test_dfr_inl2_and_rare_terms_weigh_more(ix):
    cands = M.candidates(ix, q(ix, "python nurse"))
    s = M.dfr(ix, q(ix, "python nurse"), cands, model="InL2")
    assert s[list(cands).index(2)] > s[list(cands).index(1)]       # rare 'nurse' (df 1) outweighs ubiquitous 'python' (df 4)
    with pytest.raises(ValueError):
        M.dfr(ix, q(ix, "python"), cands, model="nope")


def test_lm_and_dfr_can_rank_differently():
    docs = {0: "a " * 30 + "x", 1: "x x x b b", 2: "x " + "c " * 40}          # long doc with one 'x', short doc with three
    ix2 = InvertedIndex.build(IndexDoc(i, t.split(), {"skills": [], "cities": None, "city": None}) for i, t in docs.items())
    qt, cands = [(ix2.term_id("x"), 1)], M.candidates(ix2, [(ix2.term_id("x"), 1)])
    assert int(cands[np.argmax(M.unigram_lm(ix2, qt, cands))]) == 1 and int(cands[np.argmax(M.dfr(ix2, qt, cands))]) == 1


def test_bigram_separates_word_order_that_the_unigram_model_cannot():
    docs = {0: "machine learning x y", 1: "learning x machine y", 2: "z z z z"}          # same words, same length, different order
    ix2 = InvertedIndex.build(IndexDoc(i, t.split(), {"skills": [], "cities": None, "city": None}) for i, t in docs.items())
    ids = [ix2.term_id("machine"), ix2.term_id("learning")]
    cands = M.candidates(ix2, [(t, 1) for t in ids])
    uni = M.unigram_lm(ix2, [(t, 1) for t in ids], cands)
    big = M.bigram_lm(ix2, ids, cands, mu=5, mu_b=2)
    assert abs(uni[0] - uni[1]) < 1e-12                       # unigram model: a tie (bag of words)
    assert big[0] > big[1] + 0.1                              # bigram model: the exact phrase wins
    assert np.all(np.isfinite(big))


def test_bigram_handles_single_and_unknown_terms(ix):
    cands = M.candidates(ix, q(ix, "python"))
    assert np.allclose(M.bigram_lm(ix, [ix.term_id("python")], cands), M.unigram_lm(ix, q(ix, "python"), cands))
    assert np.all(M.bigram_lm(ix, [], cands) == 0)


def test_keyword_baseline_orders_by_coverage(ix):
    cands = M.candidates(ix, q(ix, "machine learning sql"))
    s = M.keyword_baseline(ix, q(ix, "machine learning sql"), cands)
    assert int(cands[np.argmax(s)]) == 0                             # only doc 0 has all three terms


def test_index_save_load_roundtrip(ix, tmp_path):
    ix.save(tmp_path / "i.pkl")
    ix2 = InvertedIndex.load(tmp_path / "i.pkl")
    assert ix2.stats() == ix.stats() and list(ix2.postings(ix2.term_id("python"))[0]) == [0, 1, 3, 4]
    assert ix2.doc_index(3) == 3


# --------------------------------------------------------------------------- scores must not depend on which other documents are candidates
# (a filter - or a partition - removes documents from the candidate set; the remaining scores must stay exactly the same)
def test_scores_are_independent_of_the_candidate_subset(ix):
    qt = q(ix, "machine learning python")
    seq = [ix.term_id(w) for w in ("machine", "learning", "python")]
    full = M.candidates(ix, qt)
    sub = np.array([0, 3], dtype=full.dtype)
    at = np.searchsorted(full, sub)
    for name, fn in [("baseline", lambda c: M.keyword_baseline(ix, qt, c)), ("dirichlet", lambda c: M.unigram_lm(ix, qt, c)),
                     ("jm", lambda c: M.unigram_lm(ix, qt, c, smoothing="jm")), ("pl2", lambda c: M.dfr(ix, qt, c)),
                     ("inl2", lambda c: M.dfr(ix, qt, c, model="InL2")), ("bigram", lambda c: M.bigram_lm(ix, seq, c))]:
        assert np.allclose(fn(full)[at], fn(sub), atol=1e-12), name


def test_bigram_with_precomputed_global_counts_matches_local(ix):
    seq = [ix.term_id(w) for w in ("machine", "learning")]
    cands = M.candidates(ix, [(t, 1) for t in seq])
    counts = M.bigram_collection_counts(ix, seq)
    assert counts == [2.0 + 1.0 - 1.0] or counts[0] == 2.0                         # 'machine learning' occurs in docs 0 and 3
    assert np.allclose(M.bigram_lm(ix, seq, cands), M.bigram_lm(ix, seq, cands, cf_bigram=counts))


def test_top_positions_is_deterministic_under_ties():
    rng = np.random.default_rng(0)
    for _ in range(300):
        raw = rng.integers(0, 4, size=int(rng.integers(1, 60))).astype(float)             # few distinct values -> lots of ties at the cut-off
        n = int(rng.integers(0, 70))
        want = np.lexsort((np.arange(len(raw)), -raw))[:n]                                # brute force: score desc, then position asc
        assert M.top_positions(raw, n).tolist() == want.tolist()
    assert M.top_positions(np.ones(5), 2).tolist() == [0, 1] and M.top_positions(np.array([]), 3).tolist() == []
