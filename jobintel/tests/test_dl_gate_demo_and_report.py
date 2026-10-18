"""Memory-task generator, gate statistics and the report helpers."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("keras")
from jobintel.dl import gate_demo as G
from jobintel.dl import numpy_rnn as nr


def test_memory_task_structure():
    X, y, gap = G.make_memory_task(500, T=30, seed=1)
    assert X.shape == (500, 30) and set(np.unique(y)) <= set(range(8))
    for i in range(500):
        p = int(np.flatnonzero(X[i] == G.MARKER)[0])
        assert (X[i] == G.MARKER).sum() == 1                       # exactly one marker
        assert X[i, p + 1] == y[i] + 1                             # the value token follows it
        assert gap[i] == 29 - (p + 1)
        others = np.delete(X[i], [p, p + 1])
        assert set(others) <= set(G.NOISE)                         # nothing else looks like a value
    Xm, _, gm = G.make_memory_task(300, T=50, seed=2, max_start=10)
    assert gm.min() >= 50 - 1 - 11


def test_gate_selectivity_on_a_hand_built_trace():
    T, U = 8, 4
    X = np.array([10, 11, 9, 3, 12, 13, 14, 15])                   # marker at 2, value at 3
    tr = {k: np.full((T, U), 0.5) for k in "ifgoc"}
    tr["i"][3] = 0.9                                               # the write gate opens only at the value step
    tr["c"][3:] = 1.0                                              # and the cell changes only there
    tr["c"][:3] = 0.0
    s = G.gate_selectivity(tr, X)
    assert s["marker_step"] == 2 and s["value_step"] == 3
    assert abs(s["write_i_at_value"] - 0.9) < 1e-12 and abs(s["write_i_at_noise"] - 0.5) < 1e-12
    assert s["units_with_write_delta_gt_0.3"] == U
    assert s["cell_change_at_value"] > s["cell_change_at_noise"]


def test_small_lstm_learns_the_memory_task_and_numpy_trace_matches():
    out = G.train_and_eval(kinds=("lstm",), T=12, n_train=2000, n_test=500, epochs=30, seed=0)
    assert out["result"]["models"]["lstm"]["test_accuracy"] > 0.9
    X, _, _ = G.make_memory_task(3, 12, seed=5)
    m = out["models"]["lstm"]
    tr = G.trace_example(m, X[0])
    ref = m.get_layer("recurrent")(m.get_layer("embedding")(X[:1])).numpy()[0]
    assert np.max(np.abs(tr["h_last"] - ref)) < 1e-5


def _report_module():
    spec = importlib.util.spec_from_file_location("report_dl", Path(__file__).resolve().parents[1] / "scripts" / "report_dl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_helpers():
    r = _report_module()
    assert r.mr([0.5]) == "0.500"
    assert r.mr([0.4, 0.6]) == "0.500 (0.400–0.600)"
    assert r.mr([0.4, 0.6], pct=True) == "50.0% (40.0%–60.0%)"
    assert r.mr([]) == "n/a" and r.mr([None]) == "n/a"
    t = r.table(["a", "b"], [[1, 2], [3, 4]])
    assert t.splitlines()[1] == "|---|---|" and t.splitlines()[2] == "| 1 | 2 |"
    assert r.sci(float("nan")) == "n/a" and r.sci(12345.678) == "1.23e+04"
