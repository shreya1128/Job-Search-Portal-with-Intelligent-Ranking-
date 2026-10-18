"""Vanishing / exploding gradient experiment (measured, not asserted).

Question: when the loss is computed from the LAST hidden state of a recurrent
layer, how much gradient reaches the input at time step t?  For an unrolled
network the gradient of the loss with respect to x_t is a product of
T-t Jacobians dh_{k+1}/dh_k, so it shrinks (or grows) roughly geometrically with
the distance T-t.  We measure that with ``tf.GradientTape``:

    loss      = 0.5 * || h_T ||^2          (h_T = last hidden state)
    gradient  = d loss / d x_t   for every t = 1..T
    profile_t = mean over a batch of  || d loss / d x_t ||_2

and summarise each profile by

    reach  = profile_1 / profile_T        (how much of the gradient at the last
                                           step survives back to the first step)
    rate   = least-squares slope of ln(profile) against the distance T-t
             (ln-gain per step: < 0 vanishing, ~ 0 stable, > 0 exploding)

Everything runs in float64 so that a gradient of 1e-60 is measured rather than
rounded to zero in float32.

Design choices (all reported in the output, none hidden):
* Layers use their Keras defaults (orthogonal recurrent kernel, LSTM forget-gate
  bias 1) unless a variant is named, and the recurrent kernel is multiplied by a
  scale s in {0.5, 1, 1.5, 2}: an orthogonal matrix has all singular values 1,
  so s sets the recurrent gain directly.
* Several random initialisations are used and the median is reported.
* SimpleRNN is measured with tanh (the Keras default) AND with ReLU; tanh
  saturates, which limits explosion, ReLU does not.
* LSTM is also measured with a large forget bias ("lstm_fb5") and GRU with a large
  update-gate bias ("gru_zb5"): the usual remedy that opens the additive path
  (f -> 1 / z -> 1), so the effect of that path is visible.  The Keras defaults
  (forget bias 1, update bias 0) do NOT keep the gradient alive over 100+ steps;
  that is a measured result, not an assumption.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Sequence

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

DEFAULT_LENGTHS = (10, 25, 50, 100, 200)
DEFAULT_SCALES = (0.5, 1.0, 1.5, 2.0)
KINDS = ("rnn_tanh", "rnn_relu", "lstm", "lstm_fb5", "gru", "gru_zb5")


def _tf():
    import tensorflow as tf  # noqa: WPS433 (lazy: keeps imports cheap for tests)
    return tf


def make_layer(kind: str, units: int, input_dim: int, seed: int):
    """One bare Keras recurrent layer (float64, last state only), built and seeded."""
    tf = _tf()
    import keras
    from keras import layers

    keras.utils.set_random_seed(seed)
    common = dict(units=units, dtype="float64", return_sequences=False)
    if kind == "rnn_tanh":
        layer = layers.SimpleRNN(activation="tanh", **common)
    elif kind == "rnn_relu":
        layer = layers.SimpleRNN(activation="relu", **common)
    elif kind in ("lstm", "lstm_fb5"):
        layer = layers.LSTM(**common)
    elif kind in ("gru", "gru_zb5"):
        layer = layers.GRU(reset_after=True, **common)
    else:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    layer.build((None, None, input_dim))
    if kind == "lstm_fb5":
        # forget-gate bias slice is [units:2*units] (gate order i, f, c, o)
        b = layer.cell.bias.numpy().copy()
        b[units:2 * units] = 5.0
        layer.cell.bias.assign(b)
    if kind == "gru_zb5":
        # update-gate bias slice is [0:units] of the input bias row (gate order z, r, h): z -> ~1 keeps the old state
        b = layer.cell.bias.numpy().copy()
        b[0, :units] = 5.0
        layer.cell.bias.assign(b)
    return layer


def scale_recurrent(layer, scale: float) -> None:
    """Multiply the recurrent kernel by ``scale`` (an orthogonal kernel has all singular values 1)."""
    cell = layer.cell
    cell.recurrent_kernel.assign(cell.recurrent_kernel.numpy() * float(scale))


def gradient_profile(layer, T: int, batch: int = 16, seed: int = 0) -> np.ndarray:
    """Mean over the batch of ||d loss / d x_t||_2 for t = 1..T, shape (T,)."""
    tf = _tf()
    rng = np.random.default_rng(seed)
    x = tf.constant(rng.standard_normal((batch, T, layer.cell.kernel.shape[0])), dtype=tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(x)
        h = layer(x)
        loss = 0.5 * tf.reduce_sum(tf.square(h))
    g = tape.gradient(loss, x).numpy()                  # (B, T, D)
    return np.linalg.norm(g, axis=2).mean(axis=0)       # (T,)


def summarise_profile(profile: np.ndarray) -> Dict[str, float]:
    """reach = first / last; rate = ln-gain per step of distance (see module docstring)."""
    p = np.asarray(profile, dtype=float)
    T = len(p)
    last, first = float(p[-1]), float(p[0])
    ok = p > 0
    dist = (T - 1 - np.arange(T))[ok].astype(float)     # 0 for the last step
    if ok.sum() >= 3:
        rate = float(np.polyfit(dist, np.log(p[ok]), 1)[0])
    else:
        rate = float("nan")
    reach = first / last if last > 0 else float("nan")
    return {"grad_last_step": last, "grad_first_step": first, "reach": reach,
            "log10_reach": math.log10(reach) if reach and reach > 0 else float("-inf"),
            "rate_per_step": rate}


def run_grid(kinds: Sequence[str] = KINDS, scales: Sequence[float] = DEFAULT_SCALES,
             lengths: Sequence[int] = DEFAULT_LENGTHS, units: int = 32, input_dim: int = 16,
             seeds: Sequence[int] = (0, 1, 2, 3, 4), batch: int = 16,
             keep_profile_for: int = 100) -> Dict:
    """Measure every (kind, scale, T). Returns a JSON-able dict; medians over ``seeds``."""
    rows: List[Dict] = []
    profiles: Dict[str, List[float]] = {}
    for kind in kinds:
        for s in scales:
            for T in lengths:
                per_seed = []
                prof_acc = []
                for sd in seeds:
                    layer = make_layer(kind, units, input_dim, seed=sd)
                    scale_recurrent(layer, s)
                    prof = gradient_profile(layer, T, batch=batch, seed=1000 + sd)
                    per_seed.append(summarise_profile(prof))
                    prof_acc.append(np.log(np.maximum(prof, 1e-300)))
                def med(key):
                    vals = [r[key] for r in per_seed if np.isfinite(r[key])]
                    return float(np.median(vals)) if vals else float("nan")
                rows.append({"kind": kind, "scale": s, "T": T,
                             "grad_last_step": med("grad_last_step"),
                             "grad_first_step": med("grad_first_step"),
                             "log10_reach": med("log10_reach"),
                             "rate_per_step": med("rate_per_step"),
                             "n_seeds": len(seeds)})
                if T == keep_profile_for:
                    profiles[f"{kind}|{s}"] = np.exp(np.median(np.stack(prof_acc), axis=0)).tolist()
    return {"setting": {"units": units, "input_dim": input_dim, "batch": batch, "seeds": list(seeds),
                        "loss": "0.5*||h_T||^2", "dtype": "float64",
                        "lengths": list(lengths), "scales": list(scales)},
            "rows": rows, "profiles_T%d" % keep_profile_for: profiles}


# ---------------------------------------------------------------------------
# clipping: what it does to the update when gradients explode
# ---------------------------------------------------------------------------
def weight_gradient_norm(layer, T: int, batch: int = 16, seed: int = 0) -> float:
    """Global L2 norm of d loss / d (all weights of the layer)."""
    tf = _tf()
    rng = np.random.default_rng(seed)
    x = tf.constant(rng.standard_normal((batch, T, layer.cell.kernel.shape[0])), dtype=tf.float64)
    with tf.GradientTape() as tape:
        loss = 0.5 * tf.reduce_sum(tf.square(layer(x)))
    grads = tape.gradient(loss, layer.trainable_weights)
    return float(tf.linalg.global_norm(grads).numpy())


def clipping_demo(T: int = 100, scale: float = 2.0, clip: float = 1.0, units: int = 32,
                  input_dim: int = 16, seeds: Sequence[int] = (0, 1, 2, 3, 4),
                  kinds: Sequence[str] = ("rnn_relu", "rnn_tanh", "lstm", "gru")) -> List[Dict]:
    """Weight-gradient norm before and after global-norm clipping (tf.clip_by_global_norm)."""
    tf = _tf()
    out = []
    for kind in kinds:
        before = []
        for sd in seeds:
            layer = make_layer(kind, units, input_dim, seed=sd)
            scale_recurrent(layer, scale)
            before.append(weight_gradient_norm(layer, T, seed=2000 + sd))
        med = float(np.median(before))
        after = min(med, clip)
        out.append({"kind": kind, "T": T, "scale": scale, "clip_norm": clip,
                    "median_weight_grad_norm": med, "after_clipping": after,
                    "was_clipped": med > clip})
    return out


def training_with_and_without_clipping(X: np.ndarray, y: np.ndarray, vocab_size: int, n_classes: int,
                                       scale: float = 3.0, epochs: int = 3, seed: int = 0,
                                       n_train: int = 1200, max_len: int = 128,
                                       optimizers: Sequence[str] = ("sgd", "adam"),
                                       sgd_lr: float = 0.05) -> List[Dict]:
    """Train the real SimpleRNN classifier on a subset of the REAL job data with an exploding
    recurrent gain, with and without ``clipnorm=1`` and for plain SGD and Adam.

    Adam divides by a running RMS of the gradient, which hides part of an explosion; plain SGD does
    not, so both are reported.  Returns, per run, the training loss per epoch, whether any loss was
    non-finite and the final training accuracy (fresh model, same seed and data every time).
    """
    import keras
    from jobintel.dl.models import build_model

    X = np.asarray(X)[:n_train, :max_len]
    y = np.asarray(y)[:n_train]
    results = []
    for opt_name in optimizers:
        for clip in (None, 1.0):
            keras.utils.set_random_seed(seed)
            model = build_model("rnn", vocab_size, n_classes, X.shape[1], clipnorm=clip)
            rec = model.get_layer("recurrent").cell
            rec.recurrent_kernel.assign(rec.recurrent_kernel.numpy() * float(scale))
            if opt_name == "sgd":
                model.compile(optimizer=keras.optimizers.SGD(learning_rate=sgd_lr, clipnorm=clip),
                              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            hist = model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, shuffle=True)
            losses = [float(v) for v in hist.history["loss"]]
            results.append({"optimizer": opt_name, "clipnorm": clip, "recurrent_scale": scale,
                            "train_loss_by_epoch": losses,
                            "any_nonfinite_loss": bool(any(not np.isfinite(v) for v in losses)),
                            "final_train_accuracy": float(hist.history["accuracy"][-1]),
                            "n_train": int(len(X)), "max_len": int(X.shape[1])})
    return results
