"""Gradient experiment: the measuring code must itself be correct before its numbers mean anything."""
import numpy as np
import pytest

pytest.importorskip("keras")
from jobintel.dl import gradients as gr


def test_profile_matches_finite_differences():
    """d loss / d x_t from GradientTape equals a numerical derivative (float64)."""
    tf = pytest.importorskip("tensorflow")
    layer = gr.make_layer("rnn_tanh", units=4, input_dim=3, seed=0)
    T, B = 6, 2
    rng = np.random.default_rng(0)
    x = rng.standard_normal((B, T, 3))

    def loss(xv):
        return float(0.5 * np.sum(layer(tf.constant(xv, dtype=tf.float64)).numpy() ** 2))

    xt = tf.constant(x, dtype=tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(xt)
        l = 0.5 * tf.reduce_sum(tf.square(layer(xt)))
    g = tape.gradient(l, xt).numpy()
    eps = 1e-6
    for (b, t, d) in [(0, 0, 0), (1, 3, 2), (0, 5, 1)]:
        xp, xm = x.copy(), x.copy()
        xp[b, t, d] += eps; xm[b, t, d] -= eps
        num = (loss(xp) - loss(xm)) / (2 * eps)
        assert abs(num - g[b, t, d]) < 1e-6 * max(1.0, abs(num))


def test_summarise_profile_recovers_known_geometric_decay():
    T, q = 50, 0.8
    prof = q ** (T - 1 - np.arange(T))                 # shrinks by 0.8 per step going backwards
    s = gr.summarise_profile(prof)
    assert abs(s["rate_per_step"] - np.log(q)) < 1e-9
    assert abs(s["log10_reach"] - (T - 1) * np.log10(q)) < 1e-9
    grow = gr.summarise_profile(1.3 ** (T - 1 - np.arange(T)))
    assert grow["rate_per_step"] > 0


def test_small_recurrent_gain_vanishes_and_large_gain_grows_for_relu_rnn():
    out = gr.run_grid(kinds=("rnn_relu",), scales=(0.5, 2.0), lengths=(50,), units=16, input_dim=8, seeds=(0, 1, 2), batch=8)
    rate = {r["scale"]: r["rate_per_step"] for r in out["rows"]}
    assert rate[0.5] < 0 < rate[2.0]


def test_lstm_with_large_forget_bias_reaches_further_than_default_lstm():
    out = gr.run_grid(kinds=("lstm", "lstm_fb5"), scales=(1.0,), lengths=(100,), units=16, input_dim=8, seeds=(0, 1, 2), batch=8)
    reach = {r["kind"]: r["log10_reach"] for r in out["rows"]}
    assert reach["lstm_fb5"] > reach["lstm"]


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        gr.make_layer("transformer", 4, 3, 0)


def test_clipping_demo_reports_clipping_only_when_needed():
    rows = gr.clipping_demo(T=30, scale=2.0, clip=1.0, units=8, input_dim=4, seeds=(0, 1), kinds=("rnn_relu",))
    r = rows[0]
    assert r["after_clipping"] <= 1.0 + 1e-12
    assert r["was_clipped"] == (r["median_weight_grad_norm"] > 1.0)
