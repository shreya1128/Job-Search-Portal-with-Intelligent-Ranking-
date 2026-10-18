import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS.parent))
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def add_mcf():
    return load("add_mcf")


@pytest.fixture()
def fetch():
    return load("fetch_datasets")


GOOD = {"records": 20298, "unique": 18733, "with_salary": 16078, "first_date": "2018-07-09", "last_date": "2019-06-18"}


def test_verification_accepts_the_original_numbers_and_rejects_anything_else(add_mcf):
    assert all(c[3] for c in add_mcf.check_collection(GOOD))
    for key, bad in (("records", 20000), ("unique", 18000), ("with_salary", 9000), ("first_date", "2019-01-01"), ("last_date", "2019-06-17")):
        checks = add_mcf.check_collection({**GOOD, key: bad})
        assert not all(c[3] for c in checks), key                                       # a partial / different dataset must be caught
    assert not all(c[3] for c in add_mcf.check_collection({**GOOD, "records": 0, "unique": 0, "with_salary": 0}))


def test_plan_is_idempotent_and_ordered(add_mcf):
    p = add_mcf.build_plan(add_mcf.DEFAULT_STEPS, reingest=False, have_json=True, rows_in_db=20298)
    assert [s for s, _, _ in p] == add_mcf.DEFAULT_STEPS
    run = {s: go for s, go, _ in p}
    assert run["fetch"] is False and run["ingest"] is False and run["verify"] and run["pagerank"] and run["index"]      # already loaded -> nothing is re-ingested
    fresh = {s: go for s, go, _ in add_mcf.build_plan(add_mcf.DEFAULT_STEPS, False, False, 0)}
    assert fresh["fetch"] and fresh["ingest"]
    assert {s: go for s, go, _ in add_mcf.build_plan(["ingest"], True, True, 5)}["ingest"] is True               # --reingest forces it


def _fake_repo_zip(n_jobs):
    tar = io.BytesIO()
    payload = json.dumps({"jobs": [{"job_id": i} for i in range(n_jobs)]}).encode()
    with tarfile.open(fileobj=tar, mode="w:gz") as tf:
        info = tarfile.TarInfo("mycareersfuture.json")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("repo-main/data/mycareersfuture.tar.gz", tar.getvalue())
    return zipfile.ZipFile(io.BytesIO(z.getvalue()))


def test_fetch_falls_back_when_the_original_repository_is_gone_and_checks_the_count(fetch, monkeypatch, tmp_path, capsys):
    primary, fallback = fetch.REPOS["mcf"][0], fetch.FALLBACKS["mcf"][0]
    calls = []

    def fake(repo):
        calls.append(repo)
        if repo == primary:
            raise RuntimeError("404 Client Error")
        return _fake_repo_zip(3)
    monkeypatch.setattr(fetch, "download_zip", fake)
    monkeypatch.setattr(fetch, "RAW_DIR", tmp_path)
    fetch.fetch("mcf")
    out = capsys.readouterr().out
    assert calls == [primary, fallback] and "fallback repository" in out
    assert "WARNING" in out and "different version" in out                              # 3 postings != the 20,298 the project was built on
    assert (tmp_path / "mycareersfuture" / "mycareersfuture.json").exists()


def test_fetch_fails_loudly_when_nothing_can_be_downloaded(fetch, monkeypatch):
    monkeypatch.setattr(fetch, "download_zip", lambda repo: (_ for _ in ()).throw(RuntimeError("403")))
    with pytest.raises(RuntimeError, match="no repository could be downloaded"):
        fetch.download_zip_any("mcf")


def test_an_interrupted_index_write_is_detected(add_mcf):
    assert add_mcf.index_tables_complete(16138, 16138)
    assert not add_mcf.index_tables_complete(15500, 16138)                              # the partial state seen when a run was cut off by a time limit
    assert not add_mcf.index_tables_complete(0, 0)
