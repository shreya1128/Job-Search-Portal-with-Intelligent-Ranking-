"""Case studies: generators and loaders are tested; the heavy training functions are smoke-tested with tiny budgets."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("keras")
from jobintel.dl import case_studies as C


def test_traffic_generator_shapes_labels_and_determinism():
    X, kind = C.make_traffic(400, 30, seed=3)
    assert X.shape == (400 + 4 * 30, len(C.FEATURES)) and len(kind) == len(X)
    assert set(kind) <= set(C.NORMAL_KINDS) | set(C.ATTACK_KINDS)
    assert np.isfinite(X).all() and (X >= 0).all()
    X2, kind2 = C.make_traffic(400, 30, seed=3)
    assert np.array_equal(X, X2) and np.array_equal(kind, kind2)
    assert not np.array_equal(X, C.make_traffic(400, 30, seed=4)[0])


def test_attack_types_differ_from_normal_in_the_expected_direction():
    rng = np.random.default_rng(0)
    pps = C.FEATURES.index("pkts_per_s")
    dur = C.FEATURES.index("duration_s")
    web, dos, scan = (C._flows(k, 500, rng) for k in ("web", "dos", "portscan"))
    assert np.median(dos[:, pps]) > 10 * np.median(web[:, pps])         # floods send far more packets per second
    assert np.median(scan[:, dur]) < 0.1 * np.median(web[:, dur])       # scans are very short
    with pytest.raises(ValueError):
        C._flows("unknown", 3, rng)


def test_load_cic_ids_cleans_columns_infinities_and_labels(tmp_path):
    p = tmp_path / "cic.csv"
    pd.DataFrame({" Flow Duration": [10, 20, np.inf, 40], " Total Fwd Packets": [1, 2, 3, np.nan],
                  " Label": ["BENIGN", "DoS Hulk", "BENIGN", "PortScan"]}).to_csv(p, index=False)
    X, y, cols = C.load_cic_ids(str(p))
    assert cols == ["Flow Duration", "Total Fwd Packets"]
    assert X.shape == (2, 2) and y.tolist() == [False, True]            # inf and NaN rows removed
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        C.load_cic_ids(str(p))


def test_traffic_anomaly_study_detects_the_clear_attacks():
    r = C.traffic_anomaly_study(n_normal=3000, n_attack_per_kind=80, epochs=25, seed=1)
    assert r["autoencoder"]["auc"] > 0.8
    assert 0 <= r["autoencoder"]["false_positive_rate"] <= 0.1          # threshold was the 99% validation quantile
    assert r["per_attack"]["dos"]["auc_ae"] > 0.9
    assert set(r["per_attack"]) == set(C.ATTACK_KINDS)
    assert "SYNTHETIC" in r["data"]


def test_image_and_vae_studies_run_and_report_sane_numbers():
    c = C.image_compression_study(latents=(8,), epochs=3)
    row = c["rows"][0]
    assert row["compression_ratio"] == 8 and 0 < row["mse_pca"] < 0.1 and 0 < row["mse_conv_ae"] < 0.2
    v = C.vae_study(latent=4, epochs=3, n_samples=50)
    assert v["test_kl_nats_per_image"] >= 0 and 0 <= v["generated"]["mean_top_class_probability"] <= 1
    assert sum(v["generated"]["class_histogram"]) == 50
