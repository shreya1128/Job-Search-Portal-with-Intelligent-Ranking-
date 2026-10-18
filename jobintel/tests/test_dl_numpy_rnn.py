"""The from-scratch NumPy cells must reproduce the Keras layers (same weights, same inputs)."""
import numpy as np
import pytest

keras = pytest.importorskip("keras")
from keras import layers

from jobintel.dl import numpy_rnn as nr

D, U, T = 6, 5, 9


def _x(seed=0, batch=3):
    return np.random.default_rng(seed).standard_normal((batch, T, D)).astype("float32")


def _tol(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def test_simple_rnn_matches_keras():
    layer = layers.SimpleRNN(U)
    x = _x()
    ref = layer(x).numpy()
    W, R, b = [w for w in layer.get_weights()]
    for i in range(len(x)):
        h, _ = nr.rnn_forward(x[i], W, R, b)
        assert _tol(h, ref[i]) < 1e-5


def test_birnn_matches_keras():
    layer = layers.Bidirectional(layers.SimpleRNN(U))
    x = _x(1)
    ref = layer(x).numpy()
    w = layer.get_weights()
    for i in range(len(x)):
        out = nr.birnn_forward(x[i], w[:3], w[3:])
        assert out.shape == (2 * U,)
        assert _tol(out, ref[i]) < 1e-5


def test_lstm_matches_keras_and_gates_are_in_range():
    layer = layers.LSTM(U)
    x = _x(2)
    ref = layer(x).numpy()
    W, R, b = layer.get_weights()
    for i in range(len(x)):
        tr = nr.lstm_forward(x[i], W, R, b)
        assert _tol(tr["h_last"], ref[i]) < 1e-5
        for g in "ifo":
            assert tr[g].min() >= 0 and tr[g].max() <= 1
    # Keras default: unit_forget_bias -> forget bias is 1
    assert np.allclose(b[U:2 * U], 1.0)


def test_gru_matches_keras():
    layer = layers.GRU(U, reset_after=True)
    x = _x(3)
    ref = layer(x).numpy()
    W, R, b = layer.get_weights()
    assert b.shape == (2, 3 * U)
    for i in range(len(x)):
        tr = nr.gru_forward(x[i], W, R, b)
        assert _tol(tr["h_last"], ref[i]) < 1e-5
        assert tr["z"].min() >= 0 and tr["z"].max() <= 1


def test_masking_matches_keras_and_is_padding_invariant():
    """Post-padded sequence with a mask: the state after the last REAL step must be returned."""
    inp = keras.Input((T,), dtype="int32")
    emb = layers.Embedding(20, D, mask_zero=True)
    lstm = layers.LSTM(U)
    model = keras.Model(inp, lstm(emb(inp)))
    ids = np.array([[3, 7, 9, 4, 0, 0, 0, 0, 0]], dtype="int32")        # 4 real tokens
    ref = model(ids).numpy()[0]
    table = emb.get_weights()[0]
    W, R, b = lstm.get_weights()
    mask = ids[0] != 0
    padded = nr.lstm_forward(table[ids[0]], W, R, b, mask=mask)
    assert _tol(padded["h_last"], ref) < 1e-5
    short = nr.lstm_forward(table[ids[0][:4]], W, R, b)
    assert _tol(short["h_last"], padded["h_last"]) < 1e-12


def test_lstm_cell_state_is_additive_when_forget_is_one():
    """With f=1, i=0 the cell state must be copied unchanged (the 'highway' in the equations)."""
    W = np.zeros((D, 4 * U)); R = np.zeros((U, 4 * U)); b = np.zeros(4 * U)
    b[U:2 * U] = 50.0       # f -> 1
    b[:U] = -50.0           # i -> 0
    b[3 * U:] = 50.0
    tr = nr.lstm_forward(np.ones((4, D)), W, R, b)
    assert np.allclose(tr["c"], 0.0, atol=1e-12)
    # now write something once: set i=1 at the cell candidate via a bias, then it must persist
    b2 = b.copy(); b2[:U] = 50.0; b2[2 * U:3 * U] = 0.5
    tr2 = nr.lstm_forward(np.ones((3, D)), W, R, b2)
    assert tr2["c"][2].min() > tr2["c"][0].min()          # accumulates additively


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        nr.gru_forward(np.zeros((3, D)), np.zeros((D, 3 * U)), np.zeros((U, 3 * U)), np.zeros(3 * U))
    with pytest.raises(ValueError):
        nr.lstm_forward(np.zeros((3, D)), np.zeros((D, 4 * U)), np.zeros((U, 4 * U)), np.zeros(4 * U), mask=np.ones(5))


def test_sigmoid_is_stable():
    z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    s = nr.sigmoid(z)
    assert np.all(np.isfinite(s)) and s[0] == 0.0 and s[-1] == 1.0 and abs(s[2] - 0.5) < 1e-12
