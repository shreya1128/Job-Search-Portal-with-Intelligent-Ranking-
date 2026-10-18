"""A controlled MEMORY task that makes LSTM gate behaviour visible (SYNTHETIC, for explanation only - not part of the job workflow).

Why: on real job text the gates move only slightly (measured in scripts/trace_gates.py), because job-function classification does not
need precise long-range memory.  To show what 'selective write / keep / read' means, we train on a task that REQUIRES it:

    a sequence of T random 'noise' tokens; at a random early position a MARKER token appears and the very next token is a VALUE (1..8);
    the label is that value.  The model must write the value into memory when the marker has just been seen, keep it through the noise,
    and read it at the end.

Token ids: 0 padding, 1..8 value tokens (only ever follow the marker), 9 marker, 10..17 noise.
The same task is used to compare SimpleRNN, LSTM and GRU on how far back they can remember (accuracy vs. the gap between value and end).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
MARKER, VALUES, NOISE = 9, list(range(1, 9)), list(range(10, 18))
VOCAB = 18


def make_memory_task(n: int, T: int = 30, seed: int = 0, max_start: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X (n,T) int32, y (n,) value in 0..7, gap (n,) = number of steps between the value token and the end)."""
    rng = np.random.default_rng(seed)
    X = rng.choice(NOISE, size=(n, T)).astype(np.int32)
    hi = (T - 2) if max_start is None else min(max_start, T - 2)
    start = rng.integers(0, hi, size=n)
    val = rng.integers(0, len(VALUES), size=n)
    X[np.arange(n), start] = MARKER
    X[np.arange(n), start + 1] = np.array(VALUES)[val]
    gap = (T - 1) - (start + 1)
    return X, val.astype(np.int64), gap


def build(kind: str, units: int = 16, emb: int = 8):
    import keras
    from keras import layers
    inp = keras.Input((None,), dtype="int32")
    x = layers.Embedding(VOCAB, emb, name="embedding")(inp)
    x = {"rnn": layers.SimpleRNN, "lstm": layers.LSTM, "gru": layers.GRU}[kind](units, name="recurrent")(x)
    out = layers.Dense(len(VALUES), activation="softmax")(x)
    m = keras.Model(inp, out, name=f"memory_{kind}")
    m.compile(optimizer=keras.optimizers.Adam(5e-3, clipnorm=1.0), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def train_and_eval(kinds: Sequence[str] = ("rnn", "lstm", "gru"), T: int = 30, n_train: int = 6000, n_test: int = 3000, epochs: int = 25,
                   seed: int = 0, max_start: Optional[int] = None) -> Dict:
    """Train each model on the same data; report test accuracy overall and by gap length (short / medium / long)."""
    import keras
    Xtr, ytr, _ = make_memory_task(n_train, T, seed, max_start)
    Xte, yte, gap = make_memory_task(n_test, T, seed + 1, max_start)
    lo_gap, hi_gap = int(gap.min()), int(gap.max())
    third = max(1, (hi_gap - lo_gap + 1) // 3)
    bins = [(lo_gap, lo_gap + third - 1), (lo_gap + third, lo_gap + 2 * third - 1), (lo_gap + 2 * third, hi_gap)]
    out: Dict = {"T": T, "max_marker_start": max_start, "gap_range": [lo_gap, hi_gap], "n_train": n_train, "n_test": n_test, "epochs": epochs, "chance": 1 / len(VALUES), "models": {}}
    trained = {}
    for kind in kinds:
        keras.utils.set_random_seed(seed)
        m = build(kind)
        h = m.fit(Xtr, ytr, epochs=epochs, batch_size=64, verbose=0, validation_split=0.1)
        pred = m.predict(Xte, verbose=0).argmax(axis=1)
        ok = pred == yte
        out["models"][kind] = {"test_accuracy": float(ok.mean()),
                               "by_gap": {f"{lo}-{hi}": float(ok[(gap >= lo) & (gap <= hi)].mean()) for lo, hi in bins if ((gap >= lo) & (gap <= hi)).any()},
                               "train_accuracy_last_epoch": float(h.history["accuracy"][-1])}
        trained[kind] = m
    return {"result": out, "models": trained}


def trace_example(model, X_row: np.ndarray) -> Dict[str, np.ndarray]:
    """NumPy LSTM trace (gates) of a trained memory-task LSTM on one sequence."""
    from jobintel.dl import numpy_rnn as nr
    table = np.asarray(model.get_layer("embedding").get_weights()[0])
    W, R, b = [np.asarray(w) for w in model.get_layer("recurrent").get_weights()]
    return nr.lstm_forward(table[X_row], W, R, b)


def gate_selectivity(trace: Dict[str, np.ndarray], X_row: np.ndarray) -> Dict[str, float]:
    """Average gate value at the VALUE step (just after the marker) vs at ordinary noise steps, and the cell-state change there."""
    start = int(np.flatnonzero(X_row == MARKER)[0])
    v = start + 1
    noise = np.setdiff1d(np.arange(len(X_row)), [start, v])
    c = trace["c"]
    dc = np.abs(np.diff(np.vstack([np.zeros((1, c.shape[1])), c]), axis=0)).mean(axis=1)
    # per-unit view: averaging over units can hide units that do the work
    d_write = trace["i"][v] - trace["i"][noise].mean(axis=0)          # (units,) write gate at the value step minus its usual level
    after = trace["f"][v + 1:].mean(axis=0) if v + 1 < len(X_row) else trace["f"][v]
    d_keep = after - trace["f"][noise].mean(axis=0)
    return {"unit_write_delta_max": float(d_write.max()), "units_with_write_delta_gt_0.3": int((d_write > 0.3).sum()),
            "unit_keep_delta_max": float(d_keep.max()), "n_units": int(len(d_write)), "marker_step": start, "value_step": v,
            "write_i_at_value": float(trace["i"][v].mean()), "write_i_at_noise": float(trace["i"][noise].mean()),
            "keep_f_at_value": float(trace["f"][v].mean()), "keep_f_at_noise_after": float(trace["f"][v + 1:].mean()) if v + 1 < len(X_row) else float("nan"),
            "read_o_at_last_step": float(trace["o"][-1].mean()), "read_o_at_noise_before_end": float(trace["o"][v + 1:-1].mean()) if v + 2 < len(X_row) else float("nan"),
            "cell_change_at_value": float(dc[v]), "cell_change_at_noise": float(dc[noise].mean())}
