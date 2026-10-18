"""Autoencoder case studies that are DELIBERATELY SEPARATE from the job-search workflow (syllabus: image compression, convolutional and
variational autoencoders, anomaly detection).  Everything is labelled with the data it really uses.

1. image compression       convolutional autoencoder vs PCA on scikit-learn's 8x8 handwritten digits (a small REAL image set that ships with the library)
2. variational autoencoder  the same digits; reconstruction, KL term, and a check of the images sampled from the prior
3. network-traffic anomaly  autoencoder trained on NORMAL flows only, threshold on reconstruction error.
                            DATA ARE SYNTHETIC (a generator with CIC-IDS2017-like flow features): the CIC-IDS2017 files were not available.
                            The scores therefore describe this generator, NOT real intrusion-detection performance.
                            ``load_cic_ids`` reads the real CSVs if you supply them (it has only been tested on a file in the same format).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ------------------------------------------------------------------------------------------------------------- digits
def load_digits_split(seed: int = 0, test_frac: float = 0.25):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits()
    X = (d.images / 16.0).astype("float32")                      # (1797, 8, 8) in [0, 1]
    return train_test_split(X, d.target, test_size=test_frac, random_state=seed, stratify=d.target)


def _psnr(mse: float) -> float:
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def conv_autoencoder(latent: int = 16):
    import keras
    from keras import layers
    inp = keras.Input((8, 8, 1))
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D(2)(x)                                  # 4x4
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)                                  # 2x2
    z = layers.Dense(latent, name="code")(layers.Flatten()(x))
    x = layers.Dense(2 * 2 * 32, activation="relu")(z)
    x = layers.Reshape((2, 2, 32))(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.UpSampling2D(2)(x)
    out = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(2e-3), loss="mse")
    return m


def image_compression_study(latents: Sequence[int] = (4, 8, 16), epochs: int = 150, seed: int = 0) -> Dict:
    """Convolutional AE vs PCA at the SAME code size (a 64-pixel image stored as k numbers; compression ratio 64/k)."""
    import keras
    from sklearn.decomposition import PCA
    Xtr, Xte, _, _ = load_digits_split(seed)
    flat_tr, flat_te = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    rows = []
    for k in latents:
        pca = PCA(n_components=k, random_state=seed).fit(flat_tr)
        mse_pca = float(((pca.inverse_transform(pca.transform(flat_te)) - flat_te) ** 2).mean())
        keras.utils.set_random_seed(seed)
        m = conv_autoencoder(k)
        m.fit(Xtr[..., None], Xtr[..., None], epochs=epochs, batch_size=64, verbose=0, validation_split=0.1,
              callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
        mse_ae = float(((m.predict(Xte[..., None], verbose=0)[..., 0] - Xte) ** 2).mean())
        rows.append({"code_size": k, "compression_ratio": 64 / k, "mse_conv_ae": mse_ae, "mse_pca": mse_pca,
                     "psnr_conv_ae_db": _psnr(mse_ae), "psnr_pca_db": _psnr(mse_pca), "conv_ae_params": int(m.count_params())})
    return {"data": "sklearn load_digits (1797 real 8x8 grey images; test = 25%)", "rows": rows}


# ----------------------------------------------------------------------------------------------------------------- VAE
def vae_study(latent: int = 8, epochs: int = 120, seed: int = 0, beta: float = 1.0, n_samples: int = 500) -> Dict:
    """Small VAE on the digits.  Loss = reconstruction (sum of squared error) + beta * KL(q(z|x) || N(0, I)).
    Quality of images SAMPLED from the prior is judged by a logistic-regression digit classifier trained on real digits:
    mean top-class probability and the number of distinct digits produced (a collapsed VAE would produce few)."""
    import keras
    import tensorflow as tf
    from keras import layers
    from sklearn.linear_model import LogisticRegression

    Xtr, Xte, ytr, yte = load_digits_split(seed)
    flat_tr, flat_te = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)

    class Sampling(layers.Layer):
        def call(self, inputs):
            mu, logvar = inputs
            eps = tf.random.normal(tf.shape(mu))
            return mu + tf.exp(0.5 * logvar) * eps

    class KL(layers.Layer):
        """adds the KL term as a layer loss; the reconstruction term is the model loss"""
        def call(self, inputs):
            mu, logvar = inputs
            kl = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
            self.add_loss(beta * tf.reduce_mean(kl) / 64.0)
            return inputs

    keras.utils.set_random_seed(seed)
    inp = keras.Input((64,))
    h = layers.Dense(64, activation="relu")(inp)
    mu, logvar = layers.Dense(latent, name="mu")(h), layers.Dense(latent, name="logvar")(h)
    mu, logvar = KL()([mu, logvar])
    z = Sampling()([mu, logvar])
    dec = [layers.Dense(64, activation="relu"), layers.Dense(64, activation="sigmoid")]
    out = dec[1](dec[0](z))
    vae = keras.Model(inp, out)
    vae.compile(optimizer=keras.optimizers.Adam(2e-3), loss="binary_crossentropy")      # per-pixel mean; KL scaled by 1/64 to match
    vae.fit(flat_tr, flat_tr, epochs=epochs, batch_size=64, verbose=0)
    enc = keras.Model(inp, [vae.get_layer("mu").output, vae.get_layer("logvar").output])
    mu_te, lv_te = [a for a in enc.predict(flat_te, verbose=0)]
    kl_te = float((-0.5 * (1 + lv_te - mu_te ** 2 - np.exp(lv_te)).sum(axis=1)).mean())
    recon = keras.Model(inp, out)                                                       # same graph, stochastic z
    # deterministic reconstruction from the mean code
    dz = keras.Input((latent,))
    dec_model = keras.Model(dz, dec[1](dec[0](dz)))
    rec_mean = dec_model.predict(mu_te, verbose=0)
    mse_vae = float(((rec_mean - flat_te) ** 2).mean())
    gen = dec_model.predict(np.random.default_rng(seed).standard_normal((n_samples, latent)).astype("float32"), verbose=0)
    clf = LogisticRegression(max_iter=2000).fit(flat_tr, ytr)
    p = clf.predict_proba(gen)
    return {"data": "sklearn load_digits", "latent": latent, "beta": beta, "test_mse_from_mean_code": mse_vae,
            "test_kl_nats_per_image": kl_te, "classifier_accuracy_on_real_test_digits": float(clf.score(flat_te, yte)),
            "generated": {"n": n_samples, "mean_top_class_probability": float(p.max(axis=1).mean()),
                          "distinct_digits_produced": int(len(set(p.argmax(axis=1)))),
                          "class_histogram": np.bincount(p.argmax(axis=1), minlength=10).tolist()}}


# ------------------------------------------------------------------------------------------------- network traffic
FEATURES = ["duration_s", "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes", "pkt_len_mean", "bytes_per_s", "pkts_per_s",
            "iat_mean_s", "syn_flags", "ack_flags", "psh_flags", "rst_flags", "fin_flags"]
NORMAL_KINDS = ("web", "dns", "ssh", "streaming")
ATTACK_KINDS = ("dos", "portscan", "bruteforce", "exfiltration")


def _flows(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    ln = lambda m, s: rng.lognormal(np.log(m), s, n)
    if kind == "web":
        dur, fp, bp = ln(0.6, 0.9), 2 + rng.poisson(10, n), 3 + rng.poisson(25, n)
        fb, bb = fp * ln(450, 0.4), bp * ln(900, 0.5)
        syn, psh, rst, fin = 1 + (rng.random(n) < .2), rng.poisson(4, n), (rng.random(n) < .03) * 1, 1 + (rng.random(n) < .3)
    elif kind == "dns":
        dur, fp, bp = ln(0.03, 0.8), 1 + (rng.random(n) < .05), np.ones(n)
        fb, bb = fp * ln(75, 0.2), bp * ln(130, 0.4)
        syn, psh, rst, fin = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    elif kind == "ssh":
        dur, fp, bp = ln(60, 1.0), 30 + rng.poisson(200, n), 30 + rng.poisson(220, n)
        fb, bb = fp * ln(90, 0.5), bp * ln(110, 0.5)
        syn, psh, rst, fin = np.ones(n), rng.poisson(120, n), (rng.random(n) < .02) * 1, np.ones(n)
    elif kind == "streaming":
        dur, fp, bp = ln(120, 0.7), 100 + rng.poisson(900, n), 800 + rng.poisson(5000, n)
        fb, bb = fp * ln(60, 0.3), bp * ln(1350, 0.15)
        syn, psh, rst, fin = np.ones(n), rng.poisson(40, n), (rng.random(n) < .01) * 1, np.ones(n)
    elif kind == "dos":
        dur, fp, bp = ln(8, 0.6), 3000 + rng.poisson(9000, n), (rng.random(n) < .1) * rng.poisson(3, n)
        fb, bb = fp * ln(62, 0.15), bp * 60.0
        syn, psh, rst, fin = fp * ln(0.9, 0.05), np.zeros(n), rng.poisson(2, n), np.zeros(n)
    elif kind == "portscan":
        dur, fp, bp = ln(0.002, 0.7), 1 + (rng.random(n) < .3), (rng.random(n) < .4) * 1
        fb, bb = fp * ln(44, 0.1), bp * 40.0
        syn, psh, rst, fin = fp.copy(), np.zeros(n), bp.copy(), np.zeros(n)
    elif kind == "bruteforce":       # deliberately close to ssh/web logins: short, balanced, regular
        dur, fp, bp = ln(1.6, 0.35), 12 + rng.poisson(6, n), 11 + rng.poisson(6, n)
        fb, bb = fp * ln(110, 0.25), bp * ln(130, 0.25)
        syn, psh, rst, fin = np.ones(n), rng.poisson(8, n), (rng.random(n) < .15) * 1, np.ones(n)
    elif kind == "exfiltration":
        dur, fp, bp = ln(90, 0.6), 3000 + rng.poisson(9000, n), 200 + rng.poisson(900, n)
        fb, bb = fp * ln(1380, 0.1), bp * ln(60, 0.3)
        syn, psh, rst, fin = np.ones(n), rng.poisson(30, n), np.zeros(n), np.ones(n)
    else:
        raise ValueError(kind)
    syn = np.minimum(syn, fp)                    # a flow cannot carry more SYN flags than forward packets
    pk = fp + bp
    tot = fb + bb
    return np.column_stack([dur, fp, bp, fb, bb, tot / np.maximum(pk, 1), tot / np.maximum(dur, 1e-6), pk / np.maximum(dur, 1e-6),
                            dur / np.maximum(pk - 1, 1), syn, np.clip(pk - syn, 0, None), psh, rst, fin]).astype("float64")


def make_traffic(n_normal: int, n_attack_per_kind: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """SYNTHETIC flows. Returns (X, kind) with kind in NORMAL_KINDS + ATTACK_KINDS as strings."""
    rng = np.random.default_rng(seed)
    X, kinds = [], []
    mix = rng.dirichlet([6, 8, 2, 1])
    counts = rng.multinomial(n_normal, mix)
    for k, c in zip(NORMAL_KINDS, counts):
        X.append(_flows(k, c, rng)); kinds += [k] * c
    for k in ATTACK_KINDS:
        X.append(_flows(k, n_attack_per_kind, rng)); kinds += [k] * n_attack_per_kind
    X = np.vstack(X)
    perm = rng.permutation(len(X))
    return X[perm], np.array(kinds)[perm]


def traffic_features(X: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(X, 0, None))


def load_cic_ids(path: str, max_rows: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Read a CIC-IDS2017 'MachineLearningCVE' style CSV: returns (numeric features, is_attack bool, column names).
    Column names are stripped (the files have leading spaces), +-inf and NaN rows dropped, the 'Label' column mapped BENIGN -> normal.
    Only tested on a small file with this layout, not on the real 2.8M-row release."""
    import pandas as pd
    df = pd.read_csv(path, nrows=max_rows)
    df.columns = [c.strip() for c in df.columns]
    if "Label" not in df.columns:
        raise ValueError("expected a 'Label' column (CIC-IDS2017 format)")
    y = df.pop("Label").astype(str).str.strip().str.upper() != "BENIGN"
    num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    ok = num.notna().all(axis=1)
    return num[ok].to_numpy(float), y[ok].to_numpy(bool), list(num.columns)


def traffic_anomaly_study(n_normal: int = 20000, n_attack_per_kind: int = 500, epochs: int = 60, seed: int = 0, latent: int = 4,
                          val_quantile: float = 0.99, data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict:
    """Train an autoencoder on NORMAL flows only; flag a flow as an attack when its reconstruction error exceeds the
    ``val_quantile`` quantile of the errors of held-out NORMAL validation flows (no attack is used to set the threshold).

    Reports precision / recall / F1 at that threshold and the threshold-free AUC, overall and per attack type, next to two simple
    baselines (PCA reconstruction error and Isolation Forest) fitted on the same normal data."""
    import keras
    from keras import layers
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    X, kind = data if data is not None else make_traffic(n_normal, n_attack_per_kind, seed)
    is_att = ~np.isin(kind, NORMAL_KINDS)
    F = traffic_features(X)
    rng = np.random.default_rng(seed + 1)
    norm_idx = np.flatnonzero(~is_att)
    rng.shuffle(norm_idx)
    n_tr, n_va = int(0.6 * len(norm_idx)), int(0.2 * len(norm_idx))
    tr, va, te_norm = norm_idx[:n_tr], norm_idx[n_tr:n_tr + n_va], norm_idx[n_tr + n_va:]
    te = np.concatenate([te_norm, np.flatnonzero(is_att)])
    mu, sd = F[tr].mean(axis=0), F[tr].std(axis=0) + 1e-9
    Z = (F - mu) / sd

    keras.utils.set_random_seed(seed)
    d = Z.shape[1]
    inp = keras.Input((d,))
    x = layers.Dense(16, activation="relu")(inp)
    x = layers.Dense(8, activation="relu")(x)
    code = layers.Dense(latent, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(code)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(d)(x)
    ae = keras.Model(inp, out)
    ae.compile(optimizer=keras.optimizers.Adam(2e-3), loss="mse")
    h = ae.fit(Z[tr], Z[tr], epochs=epochs, batch_size=128, verbose=0, validation_data=(Z[va], Z[va]),
               callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    err = lambda M: ((ae.predict(M, verbose=0, batch_size=2048) - M) ** 2).mean(axis=1)
    thr = float(np.quantile(err(Z[va]), val_quantile))
    e_te = err(Z[te])
    y_te, k_te = is_att[te], kind[te]
    pred = e_te > thr
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="binary", zero_division=0)
    pca = PCA(n_components=latent, random_state=seed).fit(Z[tr])
    pca_err = ((pca.inverse_transform(pca.transform(Z[te])) - Z[te]) ** 2).mean(axis=1)
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(Z[tr])
    iso_score = -iso.score_samples(Z[te])
    res = {"data": "SYNTHETIC flows (generator with CIC-IDS2017-like features)" if data is None else "user-supplied flows",
           "n_train_normal": int(len(tr)), "n_val_normal": int(len(va)), "n_test": int(len(te)), "n_test_attacks": int(y_te.sum()),
           "threshold_rule": f"{val_quantile:.0%} quantile of validation-normal reconstruction error", "threshold": thr,
           "autoencoder": {"precision": float(p), "recall": float(r), "f1": float(f), "auc": float(roc_auc_score(y_te, e_te)),
                           "false_positive_rate": float(pred[~y_te].mean()), "epochs_run": len(h.history["loss"])},
           "pca_baseline": {"auc": float(roc_auc_score(y_te, pca_err))}, "isolation_forest_baseline": {"auc": float(roc_auc_score(y_te, iso_score))},
           "per_attack": {}}
    for a in sorted(set(k_te[y_te])):
        m = (k_te == a) | ~y_te
        res["per_attack"][a] = {"n": int((k_te == a).sum()), "recall_at_threshold": float(pred[k_te == a].mean()),
                                "auc_ae": float(roc_auc_score(k_te[m] == a, e_te[m])), "auc_pca": float(roc_auc_score(k_te[m] == a, pca_err[m])),
                                "auc_isolation_forest": float(roc_auc_score(k_te[m] == a, iso_score[m]))}
    return res
