"""DL similarity: percentile ranks per space, spaces never mixed, honest 'off' states, and (when the Phase 5 artefacts exist) the real wiring."""
from pathlib import Path

import numpy as np
import pytest

from jobintel.config import DATA_DIR
from jobintel.rec.dlsim_spaces import DLSimilarity, Space, _unit


def _space(name, vectors, ids, zero=None):
    E = _unit(np.asarray(vectors, dtype=np.float32))
    return Space(name, E, {int(j): i for i, j in enumerate(ids)}, np.zeros(len(E), bool) if zero is None else np.asarray(zero))


def test_percentile_rank_is_the_share_of_rows_at_most_as_similar():
    sp = _space("skill", [[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], ids=[10, 11, 12, 13])
    s = sp.percentile_scores(_unit(np.array([[1.0, 0.0]]))[0], [10, 11, 12, 13, 99])
    assert s[0] == 1.0 and s[3] == 0.25            # most / least similar of the 4 rows
    assert s[0] > s[1] > s[2] > s[3]
    assert np.isnan(s[4])                          # job without a vector -> NaN, never 0


def test_all_zero_vectors_are_not_evidence():
    sp = _space("text", [[1, 0], [0, 0], [0.5, 0.5]], ids=[1, 2, 3], zero=[False, True, False])
    s = sp.percentile_scores(np.array([1.0, 0.0], dtype=np.float32), [1, 2, 3])
    assert np.isnan(s[1]) and np.isfinite(s[[0, 2]]).all()


def test_two_spaces_with_different_dimensions_are_never_compared():
    # text space is 2-d, skill space 3-d: a cross-space dot product would raise a shape error; separate queries cannot
    text = _space("text", [[1, 0], [0, 1]], ids=[1, 2])
    skill = _space("skill", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], ids=[2, 3, 4])
    seen = {}
    dl = DLSimilarity(spaces={"text": text, "skill": skill},
                      text_query=lambda t: np.array([1, 0], np.float32), skill_query=lambda sk: np.array([0, 1, 0], np.float32), min_text_words=3)
    out = dl.score("", [1, 2, 3, 4, 5], skills={"python": 1.0}, document="one two three four")
    assert dl.last_spaces_used == ["text", "skill"]
    assert out[0] == 1.0                           # job 1: text space (best of 2)
    assert out[1] == pytest.approx(0.5)            # job 2 has BOTH vectors: the text space is used (and its rank is 0.5), not the skill space
    assert out[2] == 1.0 and np.isfinite(out[3])   # jobs 3, 4: skill space only
    assert np.isnan(out[4])                        # job 5: no vector at all


def test_text_space_needs_a_real_document_otherwise_skill_space_only():
    text = _space("text", [[1, 0], [0, 1]], ids=[1, 2])
    skill = _space("skill", [[1, 0, 0], [0, 1, 0]], ids=[1, 2])
    dl = DLSimilarity(spaces={"text": text, "skill": skill}, text_query=lambda t: np.array([1, 0], np.float32),
                      skill_query=lambda sk: np.array([0, 1, 0], np.float32), min_text_words=30)
    out = dl.score("", [1, 2], skills={"x": 1.0}, document="a short note")
    assert dl.last_spaces_used == ["skill"]        # 3 words < 30: a skills list / short note is not the text the network was trained on
    assert out[1] > out[0]


def test_no_evidence_returns_none_not_a_placeholder():
    sp = _space("skill", [[1, 0]], ids=[1])
    dl = DLSimilarity(spaces={"skill": sp}, skill_query=lambda sk: None)
    assert dl.score("", [1], skills={"unknown skill": 1.0}) is None
    assert dl.score("", [1]) is None


def test_load_default_reasons(tmp_path):
    d = DLSimilarity.load_default(tmp_path)
    assert not d.available and "not found" in d.reason
    (tmp_path / "dl").mkdir()
    np.savez(tmp_path / "dl" / "embeddings_text.npz", source=np.array(["emscad"]), external_id=np.array(["1"]), vectors=np.ones((1, 4), np.float16), model="lstm_s43")
    d = DLSimilarity.load_default(tmp_path)                       # files exist but no index to translate (source, external_id) -> job id
    assert not d.available and "index" in d.reason

    class Ix:                                                      # an index that contains the keyed job, but no model files exist
        job_ids = np.array([7])
        meta = {"source": ["emscad"], "external_id": ["1"]}
    d = DLSimilarity.load_default(tmp_path, Ix())
    assert not d.available and "text space off" in d.reason       # reports why instead of guessing


@pytest.mark.skipif(not (DATA_DIR / "dl" / "embeddings_skills.npz").exists() or not (DATA_DIR / "index" / "job_index.pkl").exists(), reason="Phase 5 artefacts / index not built")
def test_real_artefacts_are_wired_and_rank_related_jobs_first():
    from jobintel.ir.index import InvertedIndex
    ix = InvertedIndex.load(DATA_DIR / "index" / "job_index.pkl")
    dl = DLSimilarity.load_default(DATA_DIR, ix)
    assert dl.available and set(dl.spaces) <= {"text", "skill"} and "skill" in dl.spaces
    assert dl.info["skill"]["share_of_index_documents"] > 0.5
    ids = [int(j) for j in ix.job_ids]
    s = dl.score("", ids, skills={"python": 1.0, "sql": 1.0, "tableau": 1.0, "power bi": 1.0})
    assert s is not None and np.isfinite(s).sum() > 1000
    top = np.argsort(-np.nan_to_num(s, nan=-1))[:30]
    have = [len({"python", "sql", "tableau", "power bi"} & set(ix.meta["skills"][d] or [])) for d in top]
    assert np.mean(have) >= 2.0                                    # the nearest jobs in the learned skill space share the query skills (random jobs would share ~0.1)
