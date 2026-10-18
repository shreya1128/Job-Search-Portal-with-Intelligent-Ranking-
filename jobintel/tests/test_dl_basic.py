import numpy as np
import pytest

from jobintel.dl.tokenizer import NUM, PAD, UNK, DLTokenizer, words
from jobintel.dl.train import class_weights


def test_words_lowercase_numbers_and_punctuation():
    assert words("Need 3-5 years of Python, SQL & C++ (5,00,000 LPA)!") == ["need", "<num>", "<num>", "years", "of", "python", "sql", "c++", "<num>", "lpa"]


def test_tokenizer_fit_encode_pad_truncate_and_roundtrip(tmp_path):
    texts = ["python sql python machine learning", "python java 10 years", "rare word appears once"] * 4 + ["onlyonce"]
    tok = DLTokenizer.fit(texts, vocab_size=50, min_count=3, max_len=6)
    assert tok.vocab["<pad>"] == PAD and tok.vocab["<unk>"] == UNK and tok.vocab["<num>"] == NUM
    assert "python" in tok.vocab and "onlyonce" not in tok.vocab                         # words rarer than min_count are not in the vocabulary
    X = tok.encode(["python java 10 zzzunseen", "python " * 20])
    assert X.shape == (2, 6) and X.dtype == np.int32
    assert list(X[0][:4]) == [tok.vocab["python"], tok.vocab["java"], NUM, UNK] and list(X[0][4:]) == [PAD, PAD]       # post-padding, <num>, <unk>
    assert (X[1] != PAD).all()                                                         # truncated to max_len
    tok.save(tmp_path / "t.json")
    tok2 = DLTokenizer.load(tmp_path / "t.json")
    assert tok2.vocab == tok.vocab and tok2.max_len == 6 and np.array_equal(tok2.encode(["python java 10"]), tok.encode(["python java 10"]))   # train/deploy identical


def test_vocabulary_is_fitted_on_the_given_texts_only():
    tok = DLTokenizer.fit(["alpha beta gamma"] * 5, min_count=1)
    assert "delta" not in tok.vocab and tok.encode_one("delta") == [UNK]


def test_class_weights_are_balanced():
    w = class_weights(np.array([0] * 90 + [1] * 10), 2)
    assert w[1] == pytest.approx(9 * w[0]) and sum(w[c] * n for c, n in ((0, 90), (1, 10))) == pytest.approx(100)


def test_models_shapes_masking_and_representation():
    keras = pytest.importorskip("keras")
    from jobintel.dl.models import KINDS, build_model, representation_model
    X = np.array([[5, 6, 7, 0, 0, 0, 0, 0], [8, 9, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
    for kind in KINDS:
        m = build_model(kind, vocab_size=30, n_classes=4, max_len=8, emb_dim=8, units=6)
        p = m.predict(X, verbose=0)
        assert p.shape == (2, 4) and np.allclose(p.sum(axis=1), 1, atol=1e-5)
        rep = representation_model(m).predict(X, verbose=0)
        assert rep.shape == (2, 12 if kind == "birnn" else 6)
        emb = m.get_layer("embedding")
        w = emb.get_weights()[0]
        w[0] = np.random.default_rng(0).normal(size=w.shape[1]) * 50                    # make the PADDING embedding huge ...
        emb.set_weights([w])
        assert np.allclose(m.predict(X, verbose=0), p, atol=1e-5), kind                  # ... it must not matter: the mask skips padding
