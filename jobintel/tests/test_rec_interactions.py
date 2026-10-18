import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from jobintel.rec.catalog import ItemCatalog
from jobintel.rec.interactions import EVENT_WEIGHTS, ItemKNN, SyntheticConfig, evaluate_holdout, generate, wilson
from rec_fixtures import make_family_index


@pytest.fixture(scope="module")
def cat():
    return ItemCatalog(make_family_index())


@pytest.fixture(scope="module")
def data(cat):
    return generate(cat, SyntheticConfig(n_users=150, seed=7))


def test_generated_data_is_labelled_synthetic_and_deterministic(cat, data):
    assert data.is_synthetic is True and "SYNTHETIC" in data.label and "SYNTHETIC" in data.summary()["label"]
    again = generate(cat, SyntheticConfig(n_users=150, seed=7))
    assert np.array_equal(data.doc, again.doc) and np.array_equal(data.clicked, again.clicked)
    assert not np.array_equal(data.doc, generate(cat, SyntheticConfig(n_users=150, seed=8)).doc)


def test_event_logic_and_distinct_pairs(cat, data):
    assert data.viewed.all()
    assert not (data.saved & ~data.clicked).any() and not (data.applied & ~data.clicked).any()      # save / apply only after a click
    pairs = set(zip(data.user.tolist(), data.doc.tolist()))
    assert len(pairs) == data.n_pairs                                                               # no job shown twice to a user
    assert cat.has_skills[data.doc].all()
    w = data.weights()
    assert set(np.unique(w)) <= set(EVENT_WEIGHTS.values())


def test_clicks_follow_affinity(cat, data):
    aff = np.array([cat.content_scores(data.personas[u]["skills"])[d] for u, d in zip(data.user, data.doc)])
    assert aff[data.clicked].mean() > aff[~data.clicked].mean() + 0.1


def test_itemknn_hand_example():
    # users 0-3 like items {0, 1}; users 4-7 like items {2, 3}
    users = [u for u in range(8) for _ in range(2)]
    items = [0, 1] * 4 + [2, 3] * 4
    m = ItemKNN(5, shrink=1.0).fit(users, items, [1.0] * 16)
    s = m.score({0: 1.0})
    assert int(np.argmax(s)) == 1 and s[2] == 0 and s[3] == 0
    s2 = m.score({2: 1.0})
    assert int(np.argmax(s2)) == 3


def test_cold_start_and_unknown_items_give_none():
    m = ItemKNN(5, shrink=1.0).fit([0, 0, 1, 1], [0, 1, 0, 1], [1.0] * 4)
    assert m.score({}) is None
    assert m.score({4: 1.0}) is None                      # item 4 never interacted: unknown to the model
    with pytest.raises(RuntimeError):
        ItemKNN(3).score({0: 1.0})


def test_shrinkage_discounts_single_user_similarities():
    users, items = [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 2, 3]
    lo = ItemKNN(4, shrink=0.0, min_item_count=1).fit(users, items, [1.0] * 6)
    hi = ItemKNN(4, shrink=10.0, min_item_count=1).fit(users, items, [1.0] * 6)
    assert hi.S[0, 1] < lo.S[0, 1]


def test_wilson_interval_sane():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and 0 <= lo and hi <= 1
    assert np.isnan(wilson(0, 0)[0])


def test_holdout_evaluation_runs_labels_itself_and_beats_random(cat, data):
    r = evaluate_holdout(data, cat, k=10, folds=3, seed=1)
    assert "SYNTHETIC" in r["label"] and r["users_evaluated"] > 30
    assert r["itemknn"]["hit_rate"] > r["random_expected"]                  # the machinery recovers the (synthetic) structure
    assert r["persona_oracle"]["hit_rate"] >= r["random_expected"]
    for name in ("itemknn", "popularity", "persona_oracle"):
        lo, hi = r[name]["ci95"]
        assert lo <= r[name]["hit_rate"] <= hi


def test_precision_recall_and_coverage_are_consistent(cat, data):
    r = evaluate_holdout(data, cat, k=10, folds=3, seed=1)
    for name in ("itemknn", "popularity", "persona_oracle"):
        x = r[name]
        assert x["recall@k"] == pytest.approx(x["hit_rate"]) and x["precision@k"] == pytest.approx(x["hits"] / (10 * r["users_evaluated"]))
        assert 0.0 <= x["coverage"] <= 1.0
    assert r["popularity"]["coverage"] <= r["itemknn"]["coverage"] + 1e-9 or r["popularity"]["coverage"] < 0.2      # popularity recommends the same few jobs to everyone
