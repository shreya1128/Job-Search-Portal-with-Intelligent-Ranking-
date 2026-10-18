import math
import numpy as np
import pytest

from jobintel.eval import metrics as E

RANKED = [10, 11, 12, 13, 14]
LABELS = {10: 2, 11: 0, 12: 1, 13: 0, 14: 0, 99: 2}          # 99 is relevant but was not retrieved


def test_binary_metrics_by_hand():
    assert E.precision_at_k(RANKED, LABELS, 5) == pytest.approx(2 / 5) and E.precision_at_k(RANKED, LABELS, 5, thr=2) == pytest.approx(1 / 5)
    assert E.precision_at_k(RANKED, LABELS, 10) == pytest.approx(2 / 10)               # fewer than k results: the denominator stays k
    assert E.recall_at_k(RANKED, LABELS, 5) == pytest.approx(2 / 3) and E.recall_at_k(RANKED, LABELS, 1) == pytest.approx(1 / 3)
    assert E.reciprocal_rank(RANKED, LABELS) == 1.0 and E.reciprocal_rank([11, 13, 12], LABELS) == pytest.approx(1 / 3)
    assert E.reciprocal_rank([11, 13], LABELS) == 0.0


def test_ndcg_by_hand():
    dcg = 3 / math.log2(2) + 0 + 1 / math.log2(4)                                      # grades 2,0,1,0,0
    ideal = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)                    # 2,2,1,0,0
    assert E.ndcg_at_k(RANKED, LABELS, 5) == pytest.approx(dcg / ideal)
    assert E.ndcg_at_k([99, 10, 12], LABELS, 3) == pytest.approx(1.0) or E.ndcg_at_k([99, 10, 12], LABELS, 3) > 0.99
    assert E.ndcg_at_k([11, 13], LABELS, 5) == 0.0


def test_queries_without_relevant_documents_are_undefined_not_zero():
    none = {1: 0, 2: 0}
    assert E.recall_at_k([1, 2], none, 2) is None and E.reciprocal_rank([1, 2], none) is None and E.ndcg_at_k([1, 2], none, 2) is None
    assert E.precision_at_k([1, 2], none, 2) == 0.0


def test_judged_coverage_flags_unjudged_results():
    assert E.judged_at_k([10, 11, 500, 501], LABELS, 4) == 0.5


def test_bootstrap_is_deterministic_and_null_for_identical_systems():
    a = [0.2, 0.5, 0.1, 0.9, 0.4, 0.6]
    r = E.paired_bootstrap(a, a)
    assert r["mean_diff"] == 0 and r["ci_low"] == r["ci_high"] == 0 and r["ties"] == 6 and r["wins"] == r["losses"] == 0
    b = [0.0, 0.2, 0.0, 0.5, 0.1, 0.3]
    r1, r2 = E.paired_bootstrap(a, b), E.paired_bootstrap(a, b)
    assert r1 == r2 and r1["wins"] == 6 and r1["ci_low"] > 0 and r1["p_value"] < 0.05
    rng = np.random.default_rng(1)
    x, y = rng.random(20), rng.random(20)                                              # no real difference
    assert E.paired_bootstrap(x, y)["ci_low"] < 0 < E.paired_bootstrap(x, y)["ci_high"]


def test_sign_flip_test_is_exact_and_holm_is_conservative():
    assert E.sign_flip_p(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])) == pytest.approx(2 / 64)         # all six differences positive: 2 of 64 sign patterns are this extreme
    assert E.sign_flip_p(np.array([1.0, -1.0])) == 1.0 and E.sign_flip_p(np.zeros(5)) == 1.0
    assert E.sign_flip_p(np.array([0.5, -0.1, 0.0, 0.3, 0.2])) == E.sign_flip_p(np.array([0.5, -0.1, 0.3, 0.2]))      # zero differences carry no information
    big = np.full(30, 0.2)
    assert E.sign_flip_p(big) < 1e-3                                                                 # Monte-Carlo branch
    adj = E.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06]) and all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03]))
