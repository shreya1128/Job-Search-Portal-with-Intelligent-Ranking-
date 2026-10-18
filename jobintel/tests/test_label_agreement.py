import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from jobintel.eval.agreement import agreement_report, cohen_kappa, confusion

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_kappa_known_values():
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15           # 2x2 table [[20, 5], [10, 15]] -> kappa = 0.4
    assert cohen_kappa(a, b, labels=(0, 1)) == pytest.approx(0.4, abs=1e-9)
    x = [0, 1, 2, 1, 0, 2, 2, 1]
    assert cohen_kappa(x, x) == pytest.approx(1.0) and cohen_kappa(x, x, "quadratic") == pytest.approx(1.0)
    assert cohen_kappa([1, 1, 1, 1], [1, 1, 1, 1]) != cohen_kappa([1, 1, 1, 1], [1, 1, 1, 1])        # NaN: no variation, kappa undefined (not silently 1)
    assert np.isnan(cohen_kappa([], []))
    with pytest.raises(ValueError):
        cohen_kappa([0], [1], weights="cubic")


def test_weighted_kappa_penalises_far_disagreements_and_matches_sklearn():
    a = [0, 0, 1, 1, 2, 2, 0, 1, 2, 2]
    near = [0, 1, 1, 2, 2, 1, 0, 1, 2, 1]                   # disagreements are one grade apart
    far = [2, 0, 1, 1, 0, 2, 0, 1, 2, 0]                    # some are two grades apart
    assert cohen_kappa(a, near, "quadratic") > cohen_kappa(a, far, "quadratic")
    sk = pytest.importorskip("sklearn.metrics")
    for w in (None, "linear", "quadratic"):
        assert cohen_kappa(a, far, w) == pytest.approx(sk.cohen_kappa_score(a, far, weights=w), abs=1e-9)


def test_confusion_and_report_structure():
    pairs = [("q1", 2, 2), ("q1", 1, 2), ("q1", 0, 0), ("q2", 2, 1), ("q2", 1, 1), ("q2", 0, 1), ("q3", 2, 2), ("q3", 0, 0), ("q3", 1, 1)]
    m = confusion([p[1] for p in pairs], [p[2] for p in pairs])
    assert m.sum() == 9 and m[2, 2] == 2
    r = agreement_report(pairs, n_boot=200)
    assert r["n_pairs"] == 9 and r["n_queries"] == 3 and 0 <= r["exact_agreement"] <= 1 and r["within_one_grade"] == 1.0
    assert set(r["per_query"]) == {"q1", "q2", "q3"} and "binary_thr1" in r and "binary_thr2" in r
    assert agreement_report([])["n_pairs"] == 0


def test_validation_sample_is_blank_and_comparison_script_runs_end_to_end(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    src = tmp_path / "ai.xlsx"
    wb = openpyxl.Workbook()
    wb.active.title = "Instructions"
    wb["Instructions"].append(["rules"])
    q = wb.create_sheet("Queries")
    q.append(["query_id", "query", "information_need"])
    j = wb.create_sheet("Judgments")
    j.append(["query_id", "query", "information_need", "job_id", "source", "external_id", "title", "company", "location", "experience", "key_skills", "description_snippet", "relevance", "notes"])
    for qid in ("q06", "q07"):
        q.append([qid, "x", "y"])
        for i in range(6):
            j.append([qid, "x", "y", i, "test", f"{qid}-{i}", "t", "c", "l", "", "", "", i % 3, "ai note"])
    wb.save(src)
    sys.path.insert(0, str(SCRIPTS.parent))

    def load(name):
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        return mod, spec
    import subprocess
    out = tmp_path / "sample.xlsx"
    subprocess.run([sys.executable, str(SCRIPTS / "make_label_validation_sample.py"), "--source", str(src), "--out", str(out), "--queries", "q06"], check=True, capture_output=True)
    ws = openpyxl.load_workbook(out)["Judgments"]
    rows = list(ws.iter_rows(values_only=True))[1:]
    assert len(rows) == 6 and all(r[0] == "q06" and r[12] is None and r[13] is None for r in rows)       # blank: the AI labels are not copied
    assert "VALIDATION SAMPLE" in openpyxl.load_workbook(out)["Instructions"].cell(1, 1).value
    w2 = openpyxl.load_workbook(out)
    for k, r in enumerate(w2["Judgments"].iter_rows(min_row=2)):
        r[12].value = [0, 1, 2, 1, 0, 2][k]                                                           # a 'human' who agrees on 4 of 6
    human = tmp_path / "human.xlsx"
    w2.save(human)
    res = subprocess.run([sys.executable, str(SCRIPTS / "compare_labels.py"), "--a", str(human), "--b", str(src), "--name-a", "human", "--name-b", "ai"], capture_output=True, text=True)
    assert res.returncode == 0 and "labelled in both: 6" in res.stdout and "weighted kappa" in res.stdout
    assert (tmp_path / "label_agreement.json").exists()
