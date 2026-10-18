"""The four sequence models of the syllabus on one common skeleton, so the comparison isolates the recurrent cell:

    token ids (max_len) -> Embedding(64, mask_zero) -> [ SimpleRNN | Bidirectional(SimpleRNN) | LSTM | GRU ](64) -> Dropout -> Dense(softmax)

The mask makes the recurrence skip padding, so the final state is the state after the LAST REAL token.  The vector after the recurrent layer
(`representation`) is the learned job representation reused by the ranking / recommendation pipeline."""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
from keras import layers

KINDS = ("rnn", "birnn", "lstm", "gru")
LABELS = {"rnn": "SimpleRNN", "birnn": "Bidirectional SimpleRNN", "lstm": "LSTM", "gru": "GRU"}


def recurrent_layer(kind: str, units: int, **kw):
    if kind == "rnn":
        return layers.SimpleRNN(units, name="recurrent", **kw)
    if kind == "birnn":
        return layers.Bidirectional(layers.SimpleRNN(units, **kw), name="recurrent")
    if kind == "lstm":
        return layers.LSTM(units, name="recurrent", **kw)
    if kind == "gru":
        return layers.GRU(units, name="recurrent", **kw)
    raise ValueError(f"kind must be one of {KINDS}")


def build_model(kind: str, vocab_size: int, n_classes: int, max_len: int, emb_dim: int = 64, units: int = 64, dropout: float = 0.3, lr: float = 1e-3,
                clipnorm: float | None = 1.0) -> keras.Model:
    inp = keras.Input(shape=(max_len,), dtype="int32", name="tokens")
    x = layers.Embedding(vocab_size, emb_dim, mask_zero=True, name="embedding")(inp)
    x = recurrent_layer(kind, units)(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(n_classes, activation="softmax", name="classifier")(x)
    model = keras.Model(inp, out, name=kind)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def representation_model(model: keras.Model) -> keras.Model:
    """Maps token ids to the vector produced by the recurrent layer (what the classifier sees)."""
    return keras.Model(model.input, model.get_layer("recurrent").output)
