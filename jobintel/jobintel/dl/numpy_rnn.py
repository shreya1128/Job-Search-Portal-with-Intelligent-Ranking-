"""The recurrent cells written out in plain NumPy, so every equation of the syllabus is visible.

The functions use exactly the weight layout of the Keras layers, so they can be loaded with ``layer.get_weights()`` and
are verified against Keras in ``tests/test_dl_numpy_rnn.py`` (max absolute difference < 1e-5).

Keras conventions (TF 2.x / Keras 3):

SimpleRNN      h_t = tanh(x_t W + h_{t-1} R + b)
BiRNN          forward RNN over x_1..x_T and a second RNN over x_T..x_1; output = [h_T^fwd ; h_1^bwd]

LSTM  (gate order i, f, c, o in the 4U-wide kernels)
               i_t = sigmoid(x_t W_i + h_{t-1} R_i + b_i)         input gate   : how much NEW information to write
               f_t = sigmoid(x_t W_f + h_{t-1} R_f + b_f)         forget gate  : how much of the OLD cell to keep
               g_t = tanh   (x_t W_c + h_{t-1} R_c + b_c)         candidate    : what would be written
               o_t = sigmoid(x_t W_o + h_{t-1} R_o + b_o)         output gate  : how much of the cell to read out
               c_t = f_t * c_{t-1} + i_t * g_t                    cell state   : the additive "memory highway"
               h_t = o_t * tanh(c_t)

GRU   (reset_after=True, gate order z, r, h; bias has shape (2, 3U): input bias and recurrent bias)
               z_t = sigmoid(x_t W_z + b_z  + h_{t-1} R_z + c_z)  update gate
               r_t = sigmoid(x_t W_r + b_r  + h_{t-1} R_r + c_r)  reset gate
               n_t = tanh   (x_t W_h + b_h  + r_t * (h_{t-1} R_h + c_h))
               h_t = z_t * h_{t-1} + (1 - z_t) * n_t

``mask`` (optional, shape (T,)) reproduces Keras' masking: a masked step leaves h (and c) unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable form
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _mask(mask: Optional[np.ndarray], T: int) -> np.ndarray:
    if mask is None:
        return np.ones(T, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if m.shape != (T,):
        raise ValueError(f"mask must have shape ({T},), got {m.shape}")
    return m


# ---------------------------------------------------------------- SimpleRNN / BiRNN
def rnn_forward(x: np.ndarray, W: np.ndarray, R: np.ndarray, b: np.ndarray, mask=None,
                reverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Return (h_last, H) where H has shape (T, U) in the order of the INPUT (also when reverse=True)."""
    x = np.asarray(x, float)
    T, U = x.shape[0], R.shape[0]
    m = _mask(mask, T)
    h = np.zeros(U)
    H = np.zeros((T, U))
    order = range(T - 1, -1, -1) if reverse else range(T)
    for t in order:
        if m[t]:
            h = np.tanh(x[t] @ W + h @ R + b)
        H[t] = h
    return h, H


def birnn_forward(x, fwd: Sequence[np.ndarray], bwd: Sequence[np.ndarray], mask=None) -> np.ndarray:
    """Keras Bidirectional(SimpleRNN, merge_mode='concat') with return_sequences=False -> [h_T^fwd ; h_1^bwd]."""
    hf, _ = rnn_forward(x, *fwd, mask=mask)
    hb, _ = rnn_forward(x, *bwd, mask=mask, reverse=True)
    return np.concatenate([hf, hb])


# ---------------------------------------------------------------- LSTM
def lstm_forward(x: np.ndarray, W: np.ndarray, R: np.ndarray, b: np.ndarray, mask=None) -> Dict[str, np.ndarray]:
    """Run an LSTM over x (T, D). Returns h_last plus the full traces of gates, cell state and hidden state, each (T, U)."""
    x = np.asarray(x, float)
    T, U = x.shape[0], R.shape[0]
    m = _mask(mask, T)
    h, c = np.zeros(U), np.zeros(U)
    keys = ("i", "f", "g", "o", "c", "h")
    tr = {k: np.zeros((T, U)) for k in keys}
    for t in range(T):
        if m[t]:
            z = x[t] @ W + h @ R + b
            i = sigmoid(z[:U])
            f = sigmoid(z[U:2 * U])
            g = np.tanh(z[2 * U:3 * U])
            o = sigmoid(z[3 * U:])
            c = f * c + i * g
            h = o * np.tanh(c)
            tr["i"][t], tr["f"][t], tr["g"][t], tr["o"][t] = i, f, g, o
        else:                           # padding: state carried over, gates reported as NaN (they did not run)
            for k in ("i", "f", "g", "o"):
                tr[k][t] = np.nan
        tr["c"][t], tr["h"][t] = c, h
    tr["h_last"] = h
    return tr


# ---------------------------------------------------------------- GRU
def gru_forward(x: np.ndarray, W: np.ndarray, R: np.ndarray, b: np.ndarray, mask=None) -> Dict[str, np.ndarray]:
    """GRU with reset_after=True (the Keras 2.x/3.x default). ``b`` has shape (2, 3U)."""
    x = np.asarray(x, float)
    T, U = x.shape[0], R.shape[0]
    if b.ndim != 2 or b.shape != (2, 3 * U):
        raise ValueError("expected a reset_after=True GRU bias of shape (2, 3U)")
    m = _mask(mask, T)
    b_in, b_rec = b[0], b[1]
    h = np.zeros(U)
    tr = {k: np.zeros((T, U)) for k in ("z", "r", "n", "h")}
    for t in range(T):
        if m[t]:
            xp = x[t] @ W + b_in
            hp = h @ R + b_rec
            z = sigmoid(xp[:U] + hp[:U])
            r = sigmoid(xp[U:2 * U] + hp[U:2 * U])
            n = np.tanh(xp[2 * U:] + r * hp[2 * U:])
            h = z * h + (1.0 - z) * n
            tr["z"][t], tr["r"][t], tr["n"][t] = z, r, n
        else:
            for k in ("z", "r", "n"):
                tr[k][t] = np.nan
        tr["h"][t] = h
    tr["h_last"] = h
    return tr


# ---------------------------------------------------------------- applying to a trained model on real text
def _recurrent_weights(model) -> List[np.ndarray]:
    return [np.asarray(w) for w in model.get_layer("recurrent").get_weights()]


def embed_tokens(model, ids: Sequence[int]) -> np.ndarray:
    """Look the token ids up in the model's Embedding table (no Keras forward pass)."""
    table = np.asarray(model.get_layer("embedding").get_weights()[0])
    return table[np.asarray(ids, int)]


def trace_sentence(model, kind: str, ids: Sequence[int]) -> Dict[str, np.ndarray]:
    """Run the NumPy LSTM/GRU of a TRAINED Keras model over one encoded sentence (real tokens only, no padding)."""
    x = embed_tokens(model, ids)
    w = _recurrent_weights(model)
    if kind == "lstm":
        return lstm_forward(x, *w)
    if kind == "gru":
        return gru_forward(x, *w)
    raise ValueError("kind must be 'lstm' or 'gru'")


def gate_summary(trace: Dict[str, np.ndarray], tokens: Sequence[str], kind: str) -> List[Dict]:
    """One row per token with the mean gate activation across hidden units (what the gates 'do' at that word).

    LSTM: write (input gate), keep (forget gate), read (output gate), and |delta c| = how much the cell state moved.
    GRU : update gate z (high = keep the old state), reset gate r (low = ignore the old state), and |delta h|.
    """
    rows = []
    n = len(tokens)
    if kind == "lstm":
        c = trace["c"]
        prev = np.vstack([np.zeros((1, c.shape[1])), c[:-1]])
        dc = np.abs(c - prev).mean(axis=1)
        for t in range(n):
            rows.append({"token": tokens[t], "write_i": float(trace["i"][t].mean()), "keep_f": float(trace["f"][t].mean()),
                         "read_o": float(trace["o"][t].mean()), "cell_change": float(dc[t])})
    else:
        h = trace["h"]
        prev = np.vstack([np.zeros((1, h.shape[1])), h[:-1]])
        dh = np.abs(h - prev).mean(axis=1)
        for t in range(n):
            rows.append({"token": tokens[t], "update_z": float(trace["z"][t].mean()), "reset_r": float(trace["r"][t].mean()),
                         "state_change": float(dh[t])})
    return rows
