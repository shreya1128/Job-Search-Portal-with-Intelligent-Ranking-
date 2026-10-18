"""DLEncoder with a tiny model and tokenizer written to a temporary folder (no real training needed)."""
import json

import numpy as np
import pytest

keras = pytest.importorskip("keras")
from jobintel.dl.encoder import DLEncoder, available_runs
from jobintel.dl.models import build_model
from jobintel.dl.tokenizer import DLTokenizer


@pytest.fixture()
def dl_dir(tmp_path):
    texts = ["python developer with sql and machine learning", "sales manager for retail customers", "graphic design and branding work"] * 5
    tok = DLTokenizer.fit(texts, vocab_size=200, min_count=1, max_len=12)
    (tmp_path / "results").mkdir()
    (tmp_path / "models").mkdir()
    tok.save(tmp_path / "tokenizer.json")
    (tmp_path / "classification_info.json").write_text(json.dumps({"classes": ["IT", "Sales", "Design"]}))
    for name, vl in (("lstm_s1", 0.9), ("gru_s1", 0.5), ("rnn_s1", 1.4)):
        kind = name.split("_")[0]
        keras.utils.set_random_seed(0)
        m = build_model(kind, tok.vocab_size, 3, 12, emb_dim=8, units=6)
        m.save(tmp_path / "models" / f"{name}.keras")
        (tmp_path / "results" / f"{name}.json").write_text(json.dumps({"model": kind, "val_loss": vl, "val_accuracy": 0.5}))
    (tmp_path / "results" / "baseline.json").write_text(json.dumps({"accuracy": 0.6}))
    (tmp_path / "results" / "lstm_s2.json").write_text(json.dumps({"model": "lstm", "val_loss": 0.1}))     # result without a saved model
    return tmp_path


def test_available_runs_ignores_baseline_and_runs_without_models(dl_dir):
    assert {r["name"] for r in available_runs(dl_dir)} == {"lstm_s1", "gru_s1", "rnn_s1"}


def test_default_model_is_lowest_validation_loss(dl_dir):
    assert DLEncoder.load(dl_dir).name == "gru_s1"
    assert DLEncoder.load(dl_dir, "rnn_s1").name == "rnn_s1"
    with pytest.raises(FileNotFoundError):
        DLEncoder.load(dl_dir, "lstm_s2")
    with pytest.raises(FileNotFoundError):
        DLEncoder.load(dl_dir / "nowhere")


def test_embed_is_unit_norm_deterministic_and_zero_for_empty(dl_dir):
    enc = DLEncoder.load(dl_dir)
    v = enc.embed(["python developer sql", "sales manager retail", "", "!!! ???"])
    assert v.shape == (4, enc.dim)
    assert np.allclose(np.linalg.norm(v[:2], axis=1), 1.0, atol=1e-5)
    assert np.all(v[2] == 0) and np.all(v[3] == 0)
    assert np.allclose(enc.embed(["python developer sql"]), v[:1], atol=1e-6)
    assert enc.embed([]).shape == (0, enc.dim)
    assert abs(enc.similarity("python developer sql", "python developer sql") - 1.0) < 1e-5
    with pytest.raises(TypeError):
        enc.embed("a single string")


def test_predict_category(dl_dir):
    enc = DLEncoder.load(dl_dir)
    out = enc.predict_category(["python developer sql", ""], top=2)
    assert out[0]["category"] in {"IT", "Sales", "Design"} and len(out[0]["top"]) == 2
    assert abs(sum(enc.predict_proba(["python developer sql"])[0]) - 1.0) < 1e-5
    assert out[1]["category"] is None and "no usable tokens" in out[1]["reason"]
