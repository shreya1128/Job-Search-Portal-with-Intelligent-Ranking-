"""Autoencoder utilities on small synthetic matrices (no database needed)."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

pytest.importorskip("keras")
from jobintel.dl import autoencoder as A


def toy_matrix(n=400, F=30, seed=0):
    """Two 'job families' that use different skill blocks, plus a little overlap."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, F), np.float32)
    fam = rng.integers(0, 2, n)
    for i in range(n):
        block = np.arange(0, 12) if fam[i] == 0 else np.arange(15, 27)
        X[i, rng.choice(block, size=rng.integers(5, 9), replace=False)] = 1
    return X, fam


def test_corrupt_invariants():
    X, _ = toy_matrix()
    noisy, dropped = A.corrupt(X, drop=0.3, add_max=3, rng=np.random.default_rng(1))
    assert ((dropped == 1) <= (X == 1)).all()                              # only present skills can be dropped
    assert (dropped.sum(axis=1) >= 1).all()                                # at least one dropped per job
    kept = (X == 1) & (dropped == 0)
    assert (kept.sum(axis=1) >= 1).all()                                   # and at least one kept
    added = (noisy == 1) & (X == 0)
    assert (added.sum(axis=1) <= 3).all()                                  # at most add_max random skills
    assert (noisy[kept] == 1).all() and (noisy[dropped == 1] == 0).all()
    assert 0.2 < dropped.sum() / X.sum() < 0.5                             # roughly the requested rate (>= 1 forced per job)


def test_group_split_never_splits_a_group():
    groups = np.repeat(np.arange(100), 5)
    s = A.group_split(groups, seed=3)
    df = pd.DataFrame({"g": groups, "s": s})
    assert (df.groupby("g").s.nunique() == 1).all()
    assert set(s) == {"train", "val", "test"}


def test_recovery_at_k_hand_example():
    scores = np.array([[0.9, 0.8, 0.7, 0.1, 0.0]])
    noisy = np.array([[1, 0, 0, 0, 0]], np.float32)       # skill 0 present -> excluded from ranking
    dropped = np.array([[0, 0, 1, 1, 0]], np.float32)     # targets: 2 and 3
    r = A.recovery_at_k(scores, noisy, dropped, ks=(1, 2, 3))
    # candidates ranked: 1 (0.8), 2 (0.7), 3 (0.1), 4 (0.0)
    assert r["recall@1"] == 0.0 and r["hit@1"] == 0.0
    assert r["recall@2"] == 0.5 and r["hit@2"] == 1.0
    assert r["recall@3"] == 1.0 and abs(r["precision@3"] - 2 / 3) < 1e-9


def test_baselines_and_cooccurrence_beat_random_on_structured_data():
    X, _ = toy_matrix(600)
    tr, te = X[:400], X[400:]
    noisy, dropped = A.corrupt(te, 0.3, 0, np.random.default_rng(2))
    rec = {b: A.recovery_at_k(A.baseline_scores(tr, noisy, b, np.random.default_rng(0)), noisy, dropped, (5,))["recall@5"]
           for b in ("random", "popularity", "cooc")}
    assert rec["cooc"] > rec["random"] + 0.1
    with pytest.raises(ValueError):
        A.baseline_scores(tr, noisy, "nope")


def test_knn_purity_separates_clean_classes_and_reports_chance():
    X, fam = toy_matrix(500)
    res = A.knn_purity(X[:100], fam[:100], X[100:], fam[100:], k=5)
    assert res["purity@5"] > 0.95 and res["knn_accuracy"] > 0.95
    assert 0.45 < res["chance_purity"] < 0.55
    shuffled = np.random.default_rng(0).permutation(fam[100:])
    assert A.knn_purity(X[:100], fam[:100], X[100:], shuffled, k=5)["purity@5"] < 0.7


def test_models_build_with_named_bottleneck_and_regularisation():
    for kind in A.KINDS:
        m = A.build_autoencoder(kind, 30, latent=8)
        assert m.output_shape == (None, 30)
        assert A.encoder_of(m).output_shape == (None, 8)
    reg = A.build_autoencoder("deep", 30, latent=8, regularise=True)
    noreg = A.build_autoencoder("deep", 30, latent=8, regularise=False)
    assert len(reg.losses) > 0 and len(noreg.losses) == 0                   # L2 penalties exist only when regularised
    assert any("drop" in l.name for l in reg.layers) and not any("drop" in l.name for l in noreg.layers)
    with pytest.raises(ValueError):
        A.build_autoencoder("variational", 30)


def test_denoising_training_learns_to_recover_dropped_skills():
    X, fam = toy_matrix(800)
    meta = pd.DataFrame({"job_id": np.arange(800), "source": "toy", "company_id": np.arange(800), "category": np.where(fam == 0, "A", "B"),
                         "split": ["train"] * 600 + ["val"] * 100 + ["test"] * 100})
    vocab = [{"skill_id": i, "name": f"skill{i}", "skill_type": "other", "df_train": 1} for i in range(30)]
    data = A.SkillData(X=sp.csr_matrix(X), vocab=vocab, meta=meta)
    model, hist = A.train_autoencoder("denoising", data, epochs=15, batch=64, latent=6, patience=15)
    assert hist["val_loss"][-1] < hist["val_loss"][0]
    res = A.evaluate_models({"denoising": model}, data, purity_source="toy")
    r_model = res["models"]["denoising"]["recovery"]["recall@5"]
    assert r_model > res["baselines"]["random"]["recovery"]["recall@5"] + 0.1
    assert res["models"]["denoising"]["cosine_recon_vs_clean"] > res["cosine_noisy_vs_clean"] - 0.2
    # application helper
    sug = A.suggest_skills(model, vocab, ["skill1", "skill2", "skill3", "not a skill"], top=5)
    names = [s["skill"] for s in sug if "skill" in s]
    assert len(names) == 5 and not {"skill1", "skill2", "skill3"} & set(names)
    assert any("unknown_skills" in s for s in sug)
    assert "error" in A.suggest_skills(model, vocab, ["zzz"])[0]


def test_topm_binarise_and_per_job_recall():
    scores = np.array([[0.1, 0.9, 0.5, 0.7], [0.4, 0.3, 0.2, 0.1]])
    b = A.topm_binarise(scores, np.array([2, 1]))
    assert b.tolist() == [[0, 1, 0, 1], [1, 0, 0, 0]]
    noisy = np.zeros((2, 4), np.float32)
    dropped = np.array([[0, 1, 1, 0], [0, 0, 0, 0]], np.float32)       # second job has nothing to recover
    r = A.recall_per_job(scores, noisy, dropped, k=2)
    assert r[0] == 0.5 and np.isnan(r[1])                              # top-2 = skills 1 and 3 -> one of the two dropped


def test_paired_cluster_bootstrap_detects_a_real_difference_and_a_null():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 3)
    a = rng.random(600) + 0.3
    b = rng.random(600)
    d = A.paired_cluster_bootstrap(a, b, groups, n_boot=500)
    assert d["ci95_low"] > 0 and d["n_employer_groups"] == 200
    null = A.paired_cluster_bootstrap(b, rng.permutation(b), groups, n_boot=500)
    assert null["ci95_low"] < 0 < null["ci95_high"]
    nan = np.full(600, np.nan)
    assert A.paired_cluster_bootstrap(np.where(np.arange(600) % 2 == 0, a, np.nan), b, groups, n_boot=50)["n_jobs"] == 300
