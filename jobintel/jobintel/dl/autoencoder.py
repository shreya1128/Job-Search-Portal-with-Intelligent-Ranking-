"""Autoencoders on job SKILL vectors: basic, deep, and denoising (the main model), with regularisation.

WHY SKILL VECTORS
  A job (or a resume) is a set of skills.  Represent it as a multi-hot vector x in {0,1}^F over the F most frequent vetted skills.
  Real postings and real resumes are INCOMPLETE and NOISY: advertisers forget skills, extraction misses some, a parser picks up a wrong
  one.  A denoising autoencoder is trained exactly for that situation:  corrupt x into x~ (drop some skills, add a few random ones)
  and learn to reconstruct the CLEAN x from x~.  Its output is then a score for every skill = "how likely does a job/resume with
  these skills also contain skill s", which gives (a) a cleaned skill profile, (b) suggestions of probably-missing skills, and (c) a
  low-dimensional latent code usable for similarity.

MODELS (all Keras; sigmoid outputs + binary cross-entropy because the target is multi-hot)
  basic      F -> L -> F                                  one bottleneck layer
  deep       F -> 512 -> 256 -> L -> 256 -> 512 -> F      stacked layers (+ optional regularisation)
  denoising  the deep architecture trained on corrupted input -> clean target   (MAIN)
  Regularisation = L2 weight penalty + dropout in the hidden layers (an ablation without it is part of the experiment).

EVALUATION (test jobs from employers never seen in training; every model sees the SAME corrupted inputs)
  * reconstruction BCE for clean->clean and noisy->clean
  * cosine(noisy, clean) vs cosine(reconstruction, clean)
  * recovery of the DROPPED skills: recall@k / hit-rate@k against baselines (random, popularity, co-occurrence)
  * k-NN category purity of raw vs latent representations, clean vs noisy queries (labelled jobs only)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

KINDS = ("basic", "deep", "denoising")
SQL = """SELECT j.job_id, s.name AS source, j.company_id, j.category, js.skill_id
         FROM jobs j JOIN crawler_sources s USING(source_id) JOIN job_skills js ON js.job_id = j.job_id
         WHERE s.source_type = 'dataset' AND j.duplicate_of IS NULL AND j.is_fraudulent = 0"""
SKILLS_SQL = "SELECT skill_id, name, skill_type FROM skills WHERE ambiguous = 0"


# ----------------------------------------------------------------------------------------------------------------- data
@dataclass
class SkillData:
    X: sp.csr_matrix                 # (n_jobs, F) binary float32
    vocab: List[Dict]                # [{"skill_id", "name", "skill_type", "df_train"}] in column order
    meta: pd.DataFrame               # job_id, source, company_id, category, split ('train'|'val'|'test')

    def idx(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.meta.split.to_numpy() == split)

    def dense(self, split: str) -> np.ndarray:
        return self.X[self.idx(split)].toarray().astype(np.float32)


def group_split(groups: np.ndarray, fractions=(0.7, 0.1, 0.2), seed: int = 42) -> np.ndarray:
    """Assign every GROUP (employer) to train/val/test; returns the split name per row.  An employer is never in two splits."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    cut1, cut2 = int(fractions[0] * len(uniq)), int((fractions[0] + fractions[1]) * len(uniq))
    where = {g: "train" for g in uniq[:cut1]}
    where.update({g: "val" for g in uniq[cut1:cut2]})
    where.update({g: "test" for g in uniq[cut2:]})
    return np.array([where[g] for g in groups])


def build_skill_data(engine, top_n: int = 1500, min_skills: int = 3, seed: int = 42) -> SkillData:
    """Multi-hot skill matrix from the database (real datasets only; no synthetic, no fraud, no duplicates).

    Order: raw pairs -> jobs with >= min_skills raw skills -> split BY EMPLOYER -> vocabulary = the top_n most frequent
    non-ambiguous skills of the TRAIN split only -> keep jobs that still have >= min_skills vocabulary skills."""
    pairs = pd.read_sql(SQL, engine)
    skills = pd.read_sql(SKILLS_SQL, engine).set_index("skill_id")
    pairs = pairs[pairs.skill_id.isin(skills.index)]
    n_raw = pairs.groupby("job_id").skill_id.nunique()
    ok = n_raw[n_raw >= min_skills].index
    pairs = pairs[pairs.job_id.isin(ok)]
    meta = pairs.drop_duplicates("job_id")[["job_id", "source", "company_id", "category"]].reset_index(drop=True)
    group = np.where(meta.company_id.notna(), meta.company_id.fillna(-1).astype(int), -meta.job_id.to_numpy())
    meta["split"] = group_split(group, seed=seed)
    train_ids = set(meta.job_id[meta.split == "train"])
    df_train = pairs[pairs.job_id.isin(train_ids)].groupby("skill_id").job_id.nunique().sort_values(ascending=False)
    top = df_train.index[:top_n].tolist()
    col = {sid: i for i, sid in enumerate(top)}
    pairs = pairs[pairs.skill_id.isin(col)]
    row = {jid: i for i, jid in enumerate(meta.job_id)}
    X = sp.csr_matrix((np.ones(len(pairs), np.float32), (pairs.job_id.map(row).to_numpy(), pairs.skill_id.map(col).to_numpy())),
                      shape=(len(meta), len(top)))
    X.data[:] = 1.0
    n_in = np.asarray(X.sum(axis=1)).ravel()
    keep = n_in >= min_skills
    X, meta = X[keep], meta[keep].reset_index(drop=True)
    vocab = [{"skill_id": int(s), "name": skills.loc[s, "name"], "skill_type": skills.loc[s, "skill_type"], "df_train": int(df_train[s])}
             for s in top]
    return SkillData(X=X.tocsr(), vocab=vocab, meta=meta)


# ----------------------------------------------------------------------------------------------------------- corruption
def corrupt(clean: np.ndarray, drop: float = 0.3, add_max: int = 3, rng: Optional[np.random.Generator] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Masking + salt noise on a dense multi-hot batch.

    * every present skill is dropped with probability ``drop`` (at least one skill is always dropped and at least one kept),
    * then 0..add_max random ABSENT skills are switched on (uniform over the vocabulary).
    Returns (noisy, dropped) where ``dropped`` marks the skills removed (the recovery target)."""
    rng = rng or np.random.default_rng()
    n, F = clean.shape
    present = clean > 0
    dropped = present & (rng.random((n, F)) < drop)
    for i in range(n):
        p = np.flatnonzero(present[i])
        if len(p) == 0:
            continue
        if not dropped[i].any():
            dropped[i, rng.choice(p)] = True
        if dropped[i, p].all() and len(p) > 1:
            dropped[i, rng.choice(p)] = False
    noisy = np.where(dropped, 0.0, clean).astype(np.float32)
    n_add = rng.integers(0, add_max + 1, size=n)
    for i in range(n):
        if n_add[i]:
            cand = np.flatnonzero((clean[i] == 0))
            noisy[i, rng.choice(cand, size=n_add[i], replace=False)] = 1.0
    return noisy, dropped.astype(np.float32)


# ------------------------------------------------------------------------------------------------------------- models
def build_autoencoder(kind: str, n_features: int, latent: int = 64, hidden: Sequence[int] = (512, 256), l2: float = 1e-5,
                      dropout: float = 0.2, lr: float = 1e-3, regularise: bool = True):
    """Keras autoencoder; the bottleneck layer is named 'latent'."""
    import keras
    from keras import layers, regularizers

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    reg = regularizers.l2(l2) if regularise else None
    inp = keras.Input(shape=(n_features,), name="skills")
    x = inp
    if kind == "basic":
        z = layers.Dense(latent, activation="relu", kernel_regularizer=reg, name="latent")(x)
        x = z
    else:
        for i, h in enumerate(hidden):
            x = layers.Dense(h, activation="relu", kernel_regularizer=reg, name=f"enc{i}")(x)
            if regularise and dropout:
                x = layers.Dropout(dropout, name=f"enc_drop{i}")(x)
        z = layers.Dense(latent, activation="relu", kernel_regularizer=reg, name="latent")(x)
        x = z
        for i, h in enumerate(reversed(hidden)):
            x = layers.Dense(h, activation="relu", kernel_regularizer=reg, name=f"dec{i}")(x)
            if regularise and dropout:
                x = layers.Dropout(dropout, name=f"dec_drop{i}")(x)
    out = layers.Dense(n_features, activation="sigmoid", name="reconstruction")(x)
    model = keras.Model(inp, out, name=f"ae_{kind}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy")
    return model


def encoder_of(model):
    import keras
    return keras.Model(model.input, model.get_layer("latent").output)


class _Batches:
    """Fresh corruption every epoch (denoising) or plain clean->clean batches."""
    def __init__(self, X: sp.csr_matrix, batch: int, noisy: bool, drop: float, add_max: int, seed: int):
        self.X, self.batch, self.noisy, self.drop, self.add_max = X, batch, noisy, drop, add_max
        self.rng = np.random.default_rng(seed)

    def epoch(self):
        order = self.rng.permutation(self.X.shape[0])
        for s in range(0, len(order), self.batch):
            clean = self.X[order[s:s + self.batch]].toarray().astype(np.float32)
            inp = corrupt(clean, self.drop, self.add_max, self.rng)[0] if self.noisy else clean
            yield inp, clean


def train_autoencoder(kind: str, data: SkillData, *, epochs: int = 40, batch: int = 256, patience: int = 4, latent: int = 64,
                      regularise: bool = True, drop: float = 0.3, add_max: int = 3, seed: int = 42, verbose: int = 0,
                      val_noise_seed: int = 7, noisy_training: Optional[bool] = None) -> Tuple[object, Dict]:
    """Train and return (model, history dict).  Early stopping on validation loss with the best weights restored.
    ``noisy_training`` defaults to True for 'denoising' and False for 'basic'/'deep'."""
    import keras

    keras.utils.set_random_seed(seed)
    noisy_training = (kind == "denoising") if noisy_training is None else noisy_training
    model = build_autoencoder(kind, data.X.shape[1], latent=latent, regularise=regularise)
    Xtr = data.X[data.idx("train")]
    Xva = data.dense("val")
    Va_in = corrupt(Xva, drop, add_max, np.random.default_rng(val_noise_seed))[0] if noisy_training else Xva   # fixed validation noise
    gen = _Batches(Xtr, batch, noisy_training, drop, add_max, seed)
    hist = {"loss": [], "val_loss": []}
    best, best_w, wait = np.inf, None, 0
    for ep in range(epochs):
        losses = []
        for xb, yb in gen.epoch():
            losses.append(model.train_on_batch(xb, yb))
        va = float(model.evaluate(Va_in, Xva, batch_size=1024, verbose=0))
        hist["loss"].append(float(np.mean(losses)))
        hist["val_loss"].append(va)
        if verbose:
            print(f"  [{kind}] epoch {ep + 1:2d} loss {hist['loss'][-1]:.5f} val {va:.5f}", flush=True)
        if va < best - 1e-6:
            best, best_w, wait = va, model.get_weights(), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.set_weights(best_w)
    hist["epochs_run"] = len(hist["loss"])
    hist["best_val_loss"] = best
    hist["params"] = int(model.count_params())
    return model, hist


# ------------------------------------------------------------------------------------------------------------ evaluation
def bce(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(pred, eps, 1 - eps)
    return float(-(target * np.log(p) + (1 - target) * np.log(1 - p)).mean())


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return (a * b).sum(axis=1) / np.maximum(na * nb, 1e-12)


def recovery_at_k(scores: np.ndarray, noisy: np.ndarray, dropped: np.ndarray, ks: Sequence[int] = (5, 10, 20)) -> Dict[str, float]:
    """Rank every skill NOT present in the noisy input by ``scores``; the targets are the dropped skills.

    recall@k = |top-k ∩ dropped| / |dropped|, hit@k = share of jobs with at least one dropped skill in the top k, precision@k = hits / k.
    Skills present in the noisy input (including the random ones that were added) are excluded from the ranking for every method."""
    s = np.where(noisy > 0, -np.inf, scores)
    kmax = max(ks)
    top = np.argpartition(-s, kmax - 1, axis=1)[:, :kmax]
    ordered = np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, axis=1), axis=1, kind="stable"), axis=1)
    out: Dict[str, float] = {}
    n_drop = dropped.sum(axis=1).astype(np.float64)
    valid = n_drop > 0
    for k in ks:
        hits = np.take_along_axis(dropped, ordered[:, :k], axis=1).sum(axis=1).astype(np.float64)
        out[f"recall@{k}"] = float((hits[valid] / n_drop[valid]).mean())
        out[f"hit@{k}"] = float((hits[valid] > 0).mean())
        out[f"precision@{k}"] = float((hits[valid] / k).mean())
    out["n_jobs"] = int(valid.sum())
    return out


def _ranked_candidates(scores: np.ndarray, noisy: np.ndarray, kmax: int) -> np.ndarray:
    s = np.where(noisy > 0, -np.inf, scores)
    top = np.argpartition(-s, kmax - 1, axis=1)[:, :kmax]
    return np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, axis=1), axis=1, kind="stable"), axis=1)


def recall_per_job(scores: np.ndarray, noisy: np.ndarray, dropped: np.ndarray, k: int = 10) -> np.ndarray:
    """recall@k of every job that has at least one dropped skill (NaN for the others); used for paired comparisons."""
    ordered = _ranked_candidates(scores, noisy, k)
    n_drop = dropped.sum(axis=1).astype(np.float64)
    hits = np.take_along_axis(dropped, ordered, axis=1).sum(axis=1).astype(np.float64)
    return np.where(n_drop > 0, hits / np.maximum(n_drop, 1), np.nan)


def topm_binarise(scores: np.ndarray, m: np.ndarray) -> np.ndarray:
    """0/1 vector keeping the m[i] highest-scoring skills of row i (ties broken by column order)."""
    rank = np.argsort(np.argsort(-scores, axis=1, kind="stable"), axis=1, kind="stable")
    return (rank < np.asarray(m)[:, None]).astype(np.float32)


def paired_cluster_bootstrap(a: np.ndarray, b: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Mean of a-b over jobs with a 95% percentile interval from resampling EMPLOYERS (jobs of one employer are not independent).
    Descriptive: it says how much the difference could move with a different set of employers, it is not a hypothesis test."""
    ok = ~(np.isnan(a) | np.isnan(b))
    d, g = (a - b)[ok], groups[ok]
    uniq, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return {"mean_diff": float(d.mean()), "ci95_low": float(np.percentile(boot, 2.5)), "ci95_high": float(np.percentile(boot, 97.5)),
            "n_jobs": int(ok.sum()), "n_employer_groups": int(len(uniq))}


def spurious_detection_auc(scores: np.ndarray, noisy: np.ndarray, clean: np.ndarray) -> float:
    """Denoising in the literal sense: among the skills PRESENT in the noisy input, the true ones (also in the clean job) should score higher
    than the random ones that were added.  AUC over all such skill occurrences of the test set (1.0 = perfect, 0.5 = chance)."""
    from sklearn.metrics import roc_auc_score
    mask = noisy > 0
    y = clean[mask] > 0
    if y.all() or not y.any():
        return float("nan")
    return float(roc_auc_score(y, scores[mask]))


def baseline_scores(Xtrain: np.ndarray, noisy: np.ndarray, kind: str, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Non-neural rankers for the same task.
    random      uniform scores
    popularity  document frequency in the training jobs
    cooc        sum over the present skills p of P(s | p) estimated from training co-occurrence counts"""
    n, F = noisy.shape
    if kind == "random":
        return (rng or np.random.default_rng(0)).random((n, F))
    df = Xtrain.sum(axis=0)
    if kind == "popularity":
        return np.tile(df, (n, 1))
    if kind == "cooc":
        C = Xtrain.T @ Xtrain                                  # (F, F) co-occurrence counts
        cond = C / np.maximum(df[:, None], 1.0)                # row p: P(s | p)
        return noisy @ cond
    raise ValueError(kind)


def knn_purity(Zq: np.ndarray, yq: np.ndarray, Zr: np.ndarray, yr: np.ndarray, k: int = 10) -> Dict[str, float]:
    """Cosine k-NN of each query among the reference set: mean share of the k neighbours with the query's label (purity),
    majority-vote accuracy, and the purity expected by chance (sum of squared reference class shares)."""
    def unit(a):
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    Q, R = unit(Zq), unit(Zr)
    same, vote_ok = [], []
    classes, yr_i = np.unique(yr, return_inverse=True)
    yq_i = np.searchsorted(classes, yq)
    for s in range(0, len(Q), 512):
        sim = Q[s:s + 512] @ R.T
        nn = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        lab = yr_i[nn]
        same.append((lab == yq_i[s:s + 512, None]).mean(axis=1))
        votes = np.apply_along_axis(lambda r: np.bincount(r, minlength=len(classes)).argmax(), 1, lab)
        vote_ok.append(votes == yq_i[s:s + 512])
    share = np.bincount(yr_i, minlength=len(classes)) / len(yr_i)
    return {"purity@%d" % k: float(np.concatenate(same).mean()), "knn_accuracy": float(np.concatenate(vote_ok).mean()),
            "chance_purity": float((share ** 2).sum()), "n_queries": int(len(Q)), "n_reference": int(len(R))}


def labelled_subset(data: SkillData, source: str = "emscad", min_per_class: int = 100) -> pd.DataFrame:
    """Rows of ``data.meta`` that carry a usable category label (not 'Other', classes with >= min_per_class jobs in the train split)."""
    m = data.meta
    m = m[(m.source == source) & m.category.notna() & (m.category != "Other")]
    counts = m[m.split == "train"].category.value_counts()
    return m[m.category.isin(counts[counts >= min_per_class].index)]


def evaluate_models(models: Dict[str, object], data: SkillData, *, drop: float = 0.3, add_max: int = 3, seed: int = 123, k_nn: int = 10,
                    ks: Sequence[int] = (5, 10, 20), purity_source: str = "emscad") -> Dict:
    """Everything in the module docstring, on the test split, with one shared corrupted test set."""
    Xte = data.dense("test")
    noisy, dropped = corrupt(Xte, drop, add_max, np.random.default_rng(seed))
    Xtr = data.dense("train")
    scores: Dict[str, np.ndarray] = {}
    out: Dict = {"n_test_jobs": int(len(Xte)), "noise": {"drop": drop, "add_max": add_max},
                 "cosine_noisy_vs_clean": float(cosine_rows(noisy, Xte).mean()), "models": {}, "baselines": {}}
    for name, m in models.items():
        rec_noisy = m.predict(noisy, batch_size=1024, verbose=0)
        rec_clean = m.predict(Xte, batch_size=1024, verbose=0)
        m_set = noisy.sum(axis=1).astype(int)                      # same SET SIZE as the noisy input (no oracle on the clean size)
        out["models"][name] = {"bce_clean_to_clean": bce(rec_clean, Xte), "bce_noisy_to_clean": bce(rec_noisy, Xte),
                               "cosine_probabilities_vs_clean": float(cosine_rows(rec_noisy, Xte).mean()),
                               "cosine_recon_vs_clean": float(cosine_rows(topm_binarise(rec_noisy, m_set), Xte).mean()),
                               "recovery": recovery_at_k(rec_noisy, noisy, dropped, ks)}
        scores[name] = rec_noisy
        out["models"][name]["spurious_skill_auc"] = spurious_detection_auc(rec_noisy, noisy, Xte)
    for b in ("random", "popularity", "cooc"):
        bs = baseline_scores(Xtr, noisy, b, np.random.default_rng(seed))
        scores[b] = bs
        out["baselines"][b] = {"recovery": recovery_at_k(bs, noisy, dropped, ks)}
        if b == "popularity":
            out["baselines"][b]["spurious_skill_auc"] = spurious_detection_auc(bs, noisy, Xte)
        elif b == "cooc":                       # support of a present skill from the OTHER present skills (self-term removed)
            out["baselines"][b]["spurious_skill_auc"] = spurious_detection_auc(bs - noisy, noisy, Xte)
    # paired differences in recall@10 (employer-clustered bootstrap) - is a model really better than the co-occurrence baseline?
    te_meta = data.meta.iloc[data.idx("test")]
    groups = np.where(te_meta.company_id.notna(), te_meta.company_id.fillna(-1).astype(int), -te_meta.job_id.astype(int).to_numpy())
    rec10 = {k: recall_per_job(v, noisy, dropped, 10) for k, v in scores.items()}
    for name in models:
        out["models"][name]["paired_recall@10_vs_cooc"] = paired_cluster_bootstrap(rec10[name], rec10["cooc"], groups)
        out["models"][name]["paired_recall@10_vs_popularity"] = paired_cluster_bootstrap(rec10[name], rec10["popularity"], groups)
    out["cosine_note"] = ("cosine_recon_vs_clean binarises the reconstruction to its top-m skills with m = size of the noisy input, so it is "
                          "comparable with the 0/1 noisy vector; cosine_probabilities_vs_clean uses the raw sigmoid outputs (always lower)")
    # k-NN purity
    lab = labelled_subset(data, purity_source)
    if len(lab) >= 200:
        tr_i = lab.index[lab.split == "train"].to_numpy()
        te_i = lab.index[lab.split == "test"].to_numpy()
        if len(te_i) >= 50 and len(tr_i) >= 200:
            Rtr = data.X[tr_i].toarray().astype(np.float32)
            Qte = data.X[te_i].toarray().astype(np.float32)
            Qnoisy = corrupt(Qte, drop, add_max, np.random.default_rng(seed + 1))[0]
            ytr, yte = lab.loc[tr_i, "category"].to_numpy(), lab.loc[te_i, "category"].to_numpy()
            pur = {"labelled_source": purity_source, "classes": sorted(set(ytr)), "k": k_nn,
                   "raw": {"clean_query": knn_purity(Qte, yte, Rtr, ytr, k_nn), "noisy_query": knn_purity(Qnoisy, yte, Rtr, ytr, k_nn)}}
            for name, m in models.items():
                enc = encoder_of(m)
                Ztr = enc.predict(Rtr, batch_size=1024, verbose=0)
                pur[name] = {"clean_query": knn_purity(enc.predict(Qte, batch_size=1024, verbose=0), yte, Ztr, ytr, k_nn),
                             "noisy_query": knn_purity(enc.predict(Qnoisy, batch_size=1024, verbose=0), yte, Ztr, ytr, k_nn)}
            out["knn_category_purity"] = pur
    return out


# ----------------------------------------------------------------------------------------------------------- application
def suggest_skills(model, vocab: List[Dict], skills: Sequence[str], top: int = 10) -> List[Dict]:
    """The use inside the platform: given the skills a user (or a job) HAS, rank the skills they probably also have/need.
    Unknown skill names are reported, never silently ignored."""
    index = {v["name"].lower(): i for i, v in enumerate(vocab)}
    x = np.zeros((1, len(vocab)), np.float32)
    unknown = []
    for s in skills:
        i = index.get(str(s).strip().lower())
        if i is None:
            unknown.append(s)
        else:
            x[0, i] = 1.0
    if x.sum() == 0:
        return [{"error": "none of the given skills is in the autoencoder vocabulary", "unknown": unknown}]
    p = model.predict(x, verbose=0)[0]
    p[x[0] > 0] = -1
    best = np.argsort(-p)[:top]
    res = [{"skill": vocab[i]["name"], "score": float(p[i])} for i in best]
    if unknown:
        res.append({"unknown_skills": unknown})
    return res
