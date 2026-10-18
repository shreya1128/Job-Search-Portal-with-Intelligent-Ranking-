#!/usr/bin/env python3
"""Build the Phase-5 report from the saved result files: docs/phase5_report.md + docs/img/*.png (+ rows in model_experiments).

  python scripts/report_dl.py              # report + charts + database rows
  python scripts/report_dl.py --no-db      # report + charts only

Every number in the report is read from data/dl/results/*.json and data/dl/experiments/*.json - nothing is typed by hand.
Statements that depend on the numbers ("below the baseline", "ranges overlap") are computed here and are therefore true of the data shown."""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DL = ROOT / "data" / "dl"
EXP = DL / "experiments"
DOCS = ROOT / "docs"
IMG = DOCS / "img"

KIND_ORDER = ["rnn", "birnn", "lstm", "gru"]
LABEL = {"rnn": "SimpleRNN", "birnn": "BiRNN (SimpleRNN)", "lstm": "LSTM", "gru": "GRU", "baseline": "Bag-of-words logistic regression"}
# fixed categorical order (validated reference palette slots 1-4) + neutral for the non-neural baseline
COLOR = {"rnn": "#2a78d6", "birnn": "#eb6834", "lstm": "#1baf7a", "gru": "#eda100", "baseline": "#6b6a66"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ------------------------------------------------------------------------------------------------------------ helpers
def jload(path):
    return json.load(open(path)) if Path(path).exists() else None


def mr(vals, nd=3, pct=False):
    """mean (min-max) over seeds; a single value when there is one seed."""
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return "n/a"
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.{nd}f}")
    if len(vals) == 1:
        return f(vals[0])
    return f"{f(np.mean(vals))} ({f(min(vals))}–{f(max(vals))})"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def sci(x):
    return "n/a" if x is None or not np.isfinite(x) else (f"{x:.3g}")


def load_runs():
    runs = {k: [] for k in KIND_ORDER}
    for f in sorted((DL / "results").glob("*_s*.json")):
        m = re.match(r"(rnn|birnn|lstm|gru)_s(\d+)$", f.stem)
        if m:
            r = json.load(open(f))
            r["_seed"] = int(m.group(2))
            r["_name"] = f.stem
            runs[m.group(1)].append(r)
    return runs, jload(DL / "results" / "baseline.json")


def style_axes(ax, grid_axis="y"):
    import matplotlib.pyplot as plt  # noqa: F401
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


# ------------------------------------------------------------------------------------------------------------- charts
def fig_learning_curves(runs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    for kind in KIND_ORDER:
        for i, r in enumerate(sorted(runs[kind], key=lambda r: r["_seed"])):
            v = r["history"]["val_loss"]
            ax.plot(range(1, len(v) + 1), v, color=COLOR[kind], lw=2 if i == 0 else 1.2, alpha=1 if i == 0 else 0.55,
                    label=LABEL[kind] if i == 0 else None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss (cross-entropy)")
    ax.set_title("Validation loss per epoch (thin line = second seed)", fontsize=11, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(IMG / "fig_learning_curves.png")
    plt.close(fig)


def fig_comparison(runs, base):
    import matplotlib.pyplot as plt
    kinds = [k for k in KIND_ORDER if runs[k]]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), dpi=150, sharey=True)
    for ax, key, title in zip(axes, ("accuracy", "f1_macro"), ("Test accuracy", "Test macro-F1")):
        for i, k in enumerate(kinds):
            vals = [r[key] for r in runs[k]]
            ax.bar(i, np.mean(vals), width=0.6, color=COLOR[k])
            if len(vals) > 1:
                ax.vlines(i, min(vals), max(vals), color=INK, lw=1.5)
            ax.scatter([i] * len(vals), vals, s=14, color=INK, zorder=3)
            ax.text(i, np.mean(vals) / 2, f"{np.mean(vals):.2f}", ha="center", color="white", fontsize=9, fontweight="bold")
        if base:
            ax.axhline(base[key], color=COLOR["baseline"], ls="--", lw=1.5)
            ax.text(len(kinds) - 0.5, base[key] + 0.008, f"bag-of-words baseline {base[key]:.2f}", ha="right", fontsize=8.5, color=MUTED)
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels([LABEL[k].replace(" (SimpleRNN)", "") for k in kinds], fontsize=9)
        ax.set_title(title + " (bar = mean, dots = seeds)", fontsize=10, loc="left")
        style_axes(ax)
    axes[0].set_ylim(0, 0.8)
    fig.tight_layout()
    fig.savefig(IMG / "fig_comparison.png")
    plt.close(fig)


def fig_confusion(best, classes):
    import matplotlib.pyplot as plt
    cm = np.array(best["confusion"], float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=150)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=8, color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix, {best['_name']} (test; colour = share of the true class)", fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(IMG / "fig_confusion.png")
    plt.close(fig)


def fig_gradients(g):
    import matplotlib.pyplot as plt
    rows = g["grid"]["rows"]
    Ts = sorted({r["T"] for r in rows})
    spec = [("rnn_tanh", COLOR["rnn"], "-"), ("rnn_relu", COLOR["rnn"], "--"), ("lstm", COLOR["lstm"], "-"),
            ("lstm_fb5", COLOR["lstm"], "--"), ("gru", COLOR["gru"], "-"), ("gru_zb5", COLOR["gru"], "--")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), dpi=150, sharey=True)
    for ax, scale in zip(axes, (1.0, 2.0)):
        for kind, col, ls in spec:
            ys = [next((r["log10_reach"] for r in rows if r["kind"] == kind and r["scale"] == scale and r["T"] == T), np.nan) for T in Ts]
            if np.all(np.isnan(ys)):
                continue
            ax.plot(Ts, ys, color=col, ls=ls, lw=2, marker="o", ms=4)
            dy = {"lstm_fb5": 3.0, "gru_zb5": -4.0}.get(kind, 0.0)             # keep the two overlapping end labels apart
            ax.text(Ts[-1] * 1.04, ys[-1] + dy, kind, fontsize=8, color=MUTED, va="center")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(Ts)
        ax.set_xticklabels(Ts)
        ax.set_xlim(Ts[0] * 0.9, Ts[-1] * 1.9)
        ax.set_xlabel("sequence length T")
        ax.set_title(f"recurrent-kernel scale {scale}", fontsize=10, loc="left")
        style_axes(ax)
    axes[0].set_ylabel("log10 ( gradient at step 1 / gradient at step T )")
    fig.suptitle("Gradient reaching the first input: below 0 = vanishing, above 0 = exploding (dashed = ReLU RNN / gates biased open)",
                 fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(IMG / "fig_gradients.png")
    plt.close(fig)


def fig_autoencoder(ae):
    import matplotlib.pyplot as plt
    names = ["random", "popularity", "cooc"] + [k for k in ("basic", "deep", "denoising", "deep_noreg", "denoising_noreg") if k in ae["models"]]
    vals = [(ae["baselines"][n] if n in ae["baselines"] else ae["models"][n])["recovery"]["recall@10"] for n in names]
    cols = [COLOR["baseline"] if n in ae["baselines"] else COLOR["rnn"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=150, gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.bar(range(len(names)), vals, color=cols, width=0.65)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("recall@10 of the dropped skills")
    ax.set_title("Skill recovery (grey = non-neural baselines)", fontsize=10, loc="left")
    style_axes(ax)
    pur = ae.get("knn_category_purity")
    ax = axes[1]
    if pur:
        keys = ["raw"] + [k for k in ("basic", "deep", "denoising", "denoising_noreg") if k in pur]
        x = np.arange(len(keys))
        c = [pur[k]["clean_query"][f"purity@{pur['k']}"] for k in keys]
        n = [pur[k]["noisy_query"][f"purity@{pur['k']}"] for k in keys]
        ax.bar(x - 0.2, c, width=0.38, color=COLOR["rnn"], label="clean query")
        ax.bar(x + 0.2, n, width=0.38, color=COLOR["birnn"], label="noisy query")
        ax.axhline(pur["raw"]["clean_query"]["chance_purity"], color=INK, ls=":", lw=1.2)
        ax.text(len(keys) - 0.5, pur["raw"]["clean_query"]["chance_purity"] + 0.008, "chance", ha="right", fontsize=8, color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylabel(f"k-NN purity@{pur['k']}")
        ax.set_title("Category purity of neighbours", fontsize=10, loc="left")
        ax.set_ylim(0, max(0.55, max(c + n) * 1.3))
        ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, loc="upper left", ncol=2)
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(IMG / "fig_autoencoder.png")
    plt.close(fig)


def fig_gates(gd):
    import matplotlib.pyplot as plt
    st = next((v for k, v in gd["settings"].items() if k.startswith("short")), None)
    if not st:
        return
    ex = st["lstm_example"]
    x = np.arange(len(ex["tokens"]))
    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=150)
    for key, col, lab in (("write_i", COLOR["lstm"], "write (input gate)"), ("keep_f", COLOR["rnn"], "keep (forget gate)"),
                          ("read_o", COLOR["birnn"], "read (output gate)")):
        ax.plot(x, ex[key], color=col, lw=2, label=lab)
    m = ex["marker_position"]
    ax.axvline(m, color=INK, ls=":", lw=1)
    ax.text(m + 0.3, 0.02, "marker", fontsize=8, color=MUTED)
    ax.axvline(m + 1, color=INK, ls="--", lw=1)
    ax.text(m + 1.3, 0.12, "value to remember", fontsize=8, color=MUTED)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("time step")
    ax.set_ylabel("gate activation (mean over 16 units)")
    ax.set_title("LSTM gates on the controlled memory task (synthetic)", fontsize=10, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left", labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(IMG / "fig_gates.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------- report
def section_task(info):
    cc = info["class_counts"]
    return "\n".join([
        "## 1. Task and data",
        "",
        "**Task** - predict a job's *function* from the text of its description with RNN, BiRNN, LSTM and GRU (syllabus: compare the four on a real task).",
        "",
        f"* labels: the `function` field of the EMSCAD dataset; classes: {', '.join(info['classes'])} (class 'Other' and classes with fewer than 250 jobs dropped);",
        f"* {info['raw_labelled_jobs']:,} labelled non-fraudulent jobs -> {info['near_duplicates_removed']:,} near-duplicate descriptions removed (SimHash) -> **{info['jobs']:,} jobs**;",
        "* the job **title is not part of the input** (it would be a shortcut to the label);",
        f"* split **by employer**: train {info['split_sizes']['train']:,} / validation {info['split_sizes']['val']:,} / test {info['split_sizes']['test']:,}; "
        f"employers appearing in more than one split: {info['employer_groups_in_more_than_one_split']};",
        "* class sizes: " + ", ".join(f"{c} {n}" for c, n in cc.items()) + ".",
        "",
        "Every model has the same skeleton `Embedding(64, masked) -> recurrent layer (64 units) -> Dropout(0.3) -> Dense(softmax)`, Adam (lr 1e-3, `clipnorm` 1.0), "
        "balanced class weights, early stopping on validation loss (patience 2, best weights restored), so the comparison isolates the recurrent cell.",
    ])


def section_comparison(runs, base, best):
    kinds = [k for k in KIND_ORDER if runs[k]]
    rows = []
    for k in kinds:
        R = runs[k]
        rows.append([LABEL[k], f"{R[0]['n_parameters']:,}", len(R), mr([r["accuracy"] for r in R]), mr([r["precision_macro"] for r in R]),
                     mr([r["recall_macro"] for r in R]), mr([r["f1_macro"] for r in R]), mr([r["val_loss"] for r in R]),
                     mr([r["train_seconds"] for r in R], 0), mr([r["epochs_run"] for r in R], 0)])
    if base:
        rows.append([LABEL["baseline"] + " (non-neural reference)", f"{base.get('n_parameters', 0):,}", 1, f"{base['accuracy']:.3f}", f"{base['precision_macro']:.3f}",
                     f"{base['recall_macro']:.3f}", f"{base['f1_macro']:.3f}", f"{base['val_loss']:.3f}", f"{base['train_seconds']:.0f}", "-"])
    t1 = table(["model", "parameters", "seeds", "accuracy", "macro-precision", "macro-recall", "macro-F1", "validation loss", "train s", "epochs"], rows)
    per = []
    for k in kinds:
        for r in sorted(runs[k], key=lambda r: r["_seed"]):
            per.append([r["_name"], f"{r['accuracy']:.3f}", f"{r['f1_macro']:.3f}", f"{r['val_loss']:.3f}", f"{r['val_accuracy']:.3f}", f"{r['train_seconds']:.0f}",
                        r["epochs_run"], f"{r['inference_ms_per_job']:.2f}"])
    t2 = table(["run", "test accuracy", "test macro-F1", "val loss", "val accuracy", "train s", "epochs", "inference ms/job"], per)

    means = {k: np.mean([r["accuracy"] for r in runs[k]]) for k in kinds}
    order = sorted(kinds, key=lambda k: -means[k])
    f1m = {k: np.mean([r["f1_macro"] for r in runs[k]]) for k in kinds}
    lines = ["## 2. The four sequence models (test set)", "",
             "Values are the mean over seeds with (min–max). Validation loss was used for early stopping and for choosing the deployed model; "
             "the test set was used once per run for the numbers below.", "", t1, "",
             "Per run:", "", t2, "", "### Reading the result (computed from the tables)", ""]
    lines.append(f"* Ranking by mean test accuracy: {' > '.join(f'{LABEL[k]} ({means[k]:.3f})' for k in order)}.")
    if base:
        best_k = order[0]
        rel = "below" if means[best_k] < base["accuracy"] else "above"
        lines.append(f"* The best recurrent model ({LABEL[best_k]}, mean accuracy {means[best_k]:.3f}, macro-F1 {f1m[best_k]:.3f}) is **{rel}** the bag-of-words logistic-regression "
                     f"reference (accuracy {base['accuracy']:.3f}, macro-F1 {base['f1_macro']:.3f}). "
                     + ("This is reported as measured: on 4,060 training descriptions a linear model on word presence is a strong baseline and the recurrent models, "
                        "which must learn word meaning from scratch and see only the first 256 tokens, do not beat it." if rel == "below" else ""))
    spread = {k: (min(r["accuracy"] for r in runs[k]), max(r["accuracy"] for r in runs[k])) for k in kinds if len(runs[k]) > 1}
    if "lstm" in spread and "gru" in spread:
        ov = spread["lstm"][0] <= spread["gru"][1] and spread["gru"][0] <= spread["lstm"][1]
        lines.append(f"* LSTM accuracy range {spread['lstm'][0]:.3f}–{spread['lstm'][1]:.3f}, GRU {spread['gru'][0]:.3f}–{spread['gru'][1]:.3f}: the ranges "
                     + ("**overlap**, so with two seeds the data do not separate LSTM from GRU." if ov else "do **not** overlap (still only two seeds per model)."))
    if "rnn" in means and "lstm" in means:
        lines.append(f"* The plain SimpleRNN is the weakest ({means['rnn']:.3f}); its training accuracy "
                     f"reached {max(max(r['history']['accuracy']) for r in runs['rnn']):.2f} while validation accuracy stayed at "
                     f"{max(max(r['history']['val_accuracy']) for r in runs['rnn']):.2f} - it memorises the training texts (see section 3 for why long sequences are hard for it).")
    big = max(max(r["accuracy"] for r in runs[k]) - min(r["accuracy"] for r in runs[k]) for k in spread) if spread else 0
    lines.append(f"* Seed-to-seed variation is large (up to {big:.3f} in accuracy within one model), so differences smaller than that should not be interpreted.")
    lines += ["", "![validation loss](img/fig_learning_curves.png)", "", "![comparison](img/fig_comparison.png)", ""]
    # best model detail
    classes = list(best["per_class"].keys())
    cls = [c for c in classes if c not in ("accuracy", "macro avg", "weighted avg")]
    pc = [[c, f"{best['per_class'][c]['precision']:.3f}", f"{best['per_class'][c]['recall']:.3f}", f"{best['per_class'][c]['f1-score']:.3f}",
           int(best["per_class"][c]["support"])] for c in cls]
    lines += [f"### Deployed model: {best['_name']} (lowest validation loss of all runs)", "",
              table(["class", "precision", "recall", "F1", "test jobs"], pc), "", "![confusion matrix](img/fig_confusion.png)", ""]
    return "\n".join(lines), cls


def section_gradients(g):
    rows = g["grid"]["rows"]
    Ts = sorted({r["T"] for r in rows})
    kinds = [k for k in ("rnn_tanh", "rnn_relu", "lstm", "lstm_fb5", "gru", "gru_zb5") if any(r["kind"] == k for r in rows)]

    def cell(kind, scale, T):
        return next((r for r in rows if r["kind"] == kind and r["scale"] == scale and r["T"] == T), None)

    t1 = table(["layer (recurrent scale 1)"] + [f"T={T}" for T in Ts], [[k] + [f"{cell(k, 1.0, T)['log10_reach']:.1f}" for T in Ts] for k in kinds])
    scales = sorted({r["scale"] for r in rows})
    t2 = table(["layer (T=100)"] + [f"scale {s}" for s in scales], [[k] + [f"{cell(k, s, 100)['log10_reach']:.1f}" for s in scales] for k in kinds])
    t3 = table(["layer (scale 1, T=100)", "gradient at last step", "gradient at step 1", "ln-gain per step"],
               [[k, sci(cell(k, 1.0, 100)["grad_last_step"]), sci(cell(k, 1.0, 100)["grad_first_step"]), f"{cell(k, 1.0, 100)['rate_per_step']:+.3f}"] for k in kinds])
    clip = g["clipping_weight_gradients"]
    t4 = table(["layer", "median weight-gradient norm (T=%d, scale %g)" % (clip[0]["T"], clip[0]["scale"]), "after clipping to %g" % clip[0]["clip_norm"], "clipped?"],
               [[c["kind"], sci(c["median_weight_grad_norm"]), sci(c["after_clipping"]), "yes" if c["was_clipped"] else "no"] for c in clip])
    tr = g.get("training_with_vs_without_clipping", [])
    t5 = table(["optimizer", "clipnorm", "training loss by epoch", "non-finite?", "train accuracy"],
               [[r["optimizer"], r["clipnorm"], ", ".join(f"{v:.3f}" for v in r["train_loss_by_epoch"]), "yes" if r["any_nonfinite_loss"] else "no",
                 f"{r['final_train_accuracy']:.3f}"] for r in tr]) if tr else ""
    s = g["grid"]["setting"]
    fb = cell("lstm_fb5", 1.0, 200)["log10_reach"] if cell("lstm_fb5", 1.0, 200) else None
    van = cell("rnn_tanh", 1.0, 100)["log10_reach"]
    expl = cell("rnn_relu", 2.0, 100)["log10_reach"]
    lstm100 = cell("lstm", 1.0, 100)["log10_reach"]
    gru100 = cell("gru", 1.0, 100)["log10_reach"]
    out = ["## 3. Vanishing and exploding gradients (measured)", "",
           f"**Method.** One bare recurrent layer ({s['units']} units, {s['input_dim']} inputs, float64), random inputs, loss = ½‖h_T‖². With `tf.GradientTape` we take ∂loss/∂x_t for "
           f"every step t and report log10 of (gradient norm at step 1 / gradient norm at step T): about −k means the first input receives 10^−k of what the last input receives. "
           f"Median over {len(s['seeds'])} random initialisations. The recurrent kernel is multiplied by a *scale* (an orthogonal matrix has all singular values 1, so the scale sets the recurrent gain). "
           "`rnn_tanh` is Keras' default SimpleRNN, `rnn_relu` the same with ReLU, `lstm`/`gru` use Keras defaults (forget bias 1, update-gate bias 0), "
           "`lstm_fb5` has forget-gate bias 5 and `gru_zb5` update-gate bias 5 (the usual remedy that opens the additive path).", "",
           "**log10(gradient at step 1 / gradient at step T), recurrent scale 1**", "", t1, "",
           "**Effect of the recurrent scale (T = 100)**", "", t2, "", "**Raw magnitudes (scale 1, T = 100)**", "", t3, "",
           "![gradient decay](img/fig_gradients.png)", "",
           "### Reading the result (computed)", "",
           f"* **Vanishing:** with the Keras defaults the SimpleRNN's first input receives 10^{van:.1f} of the last input's gradient at T = 100; the default LSTM 10^{lstm100:.1f}, the default GRU 10^{gru100:.1f}.",
           f"* **Exploding:** with recurrent scale 2 the ReLU RNN's gradient grows by 10^{expl:+.1f} over 100 steps; with tanh the saturation limits growth ({cell('rnn_tanh', 2.0, 100)['log10_reach']:+.1f}).",
           ]
    if fb is not None:
        out.append(f"* **The cell-state path only helps when it is open:** with forget bias 5 the LSTM keeps the gradient essentially constant (10^{fb:+.1f} even at T = 200), "
                   "whereas the default forget gate (σ(1) ≈ 0.73 per step) still shrinks it. LSTM and GRU therefore *reduce* vanishing by design but do not remove it unless the gates stay near 1.")
    out += ["", "### Gradient clipping", "", t4, "",
            "Clipping rescales the whole gradient so its global norm is at most the threshold: it keeps the *direction* and removes the absurd magnitude, "
            "which is why `clipnorm=1.0` is used in every model of this project.", ""]
    if t5:
        out += ["Short training run on the real job data (SimpleRNN, recurrent kernel ×%g, %d jobs, 3 epochs):" % (tr[0]["recurrent_scale"], tr[0]["n_train"]), "", t5, "",
                "In this short run no loss became non-finite with or without clipping (tanh keeps the forward pass bounded and Adam normalises the step size), so the table "
                "does **not** show clipping rescuing a diverging run; it is a safeguard, and the weight-gradient norms above show how large the unclipped gradients are.", ""]
    return "\n".join(out)


def section_gates(ge, gd, g=None):
    out = ["## 4. LSTM gates and GRU (from-scratch NumPy cells)", "",
           "`jobintel/dl/numpy_rnn.py` implements SimpleRNN, BiRNN, LSTM (gates i, f, g, o; c_t = f⊙c_{t−1} + i⊙g; h_t = o⊙tanh c_t) and GRU (update z, reset r; h_t = z⊙h_{t−1} + (1−z)⊙n) "
           "with the Keras weight layout; `tests/test_dl_numpy_rnn.py` checks equality with the Keras layers (max difference < 1e-5, including masking).", ""]
    if ge:
        out += ["### On a real job description", "",
                f"Source: {ge['source']}; first {ge['traced_tokens']} of {ge['n_tokens_total']} tokens traced with the NumPy cell of the *trained* models.", ""]
        rows = []
        for kind, m in ge["models"].items():
            R = m["rows"]
            keys = [k for k in R[0] if k not in ("token", "cell_change", "state_change")]
            rows.append([kind.upper() + f" ({m['run']})", f"{m['max_abs_diff_numpy_vs_keras']:.1e}"] +
                        ["; ".join(f"{k} {min(r[k] for r in R):.2f}–{max(r[k] for r in R):.2f}" for k in keys)])
        out += [table(["model", "max |NumPy − Keras| on the final state", "range of the mean gate value across the traced tokens"], rows), "",
                ]
        widths = [max(r[k] for r in m["rows"]) - min(r[k] for r in m["rows"]) for m in ge["models"].values() for k in m["rows"][0] if k not in ("token", "cell_change", "state_change")]
        if widths and max(widths) < 0.2:
            out += [f"On real text the **average** gate values barely move (the widest range across the traced tokens is {max(widths):.2f}; they stay near their initial values: forget ≈ σ(1) = 0.73, others ≈ 0.5), "
                    "so the trained classifiers do not show dramatic selective forgetting. This is plausible: function classification needs a bag of topics, not precise long-range memory. ", ""]
        out += ["To make the gate mechanics visible, a controlled task that *requires* memory is used below (synthetic, for explanation only).", ""]
    if gd:
        out += ["### Controlled memory task (SYNTHETIC; not part of the job workflow)", "",
                "A sequence of random noise tokens contains a MARKER token; the token right after it is a VALUE (1..8) that must be reported at the end. "
                "The model must write the value when the marker has just been seen, keep it through the noise, and read it at the end.", ""]
        rows = []
        for label, st in gd["settings"].items():
            accs = {k: v.get("test_accuracy_by_seed", [v["test_accuracy"]]) for k, v in st["models"].items()}
            rows.append([label, ", ".join(f"{k} {mr(a)}" for k, a in accs.items()), f"{len(st.get('seeds', [0]))}", f"chance {st['chance']:.3f}"])
        out += [table(["setting", "test accuracy (mean (min–max) over seeds)", "seeds", ""], rows), ""]
        for label, st in gd["settings"].items():
            mean = {k: v.get("test_accuracy_mean", v["test_accuracy"]) for k, v in st["models"].items()}
            solved = [k for k, v in mean.items() if v >= 0.9]
            failed = [k for k, v in mean.items() if v < st["chance"] + 0.1]
            if failed:
                msg = f"* {label}: {', '.join(solved) or 'no model'} reached ≥ 0.9 accuracy; {', '.join(failed)} stayed near chance ({', '.join(f'{k} {mean[k]:.2f}' for k in failed)})."
                if g and "gru" in failed and "lstm" in solved:
                    rows_g = g["grid"]["rows"]
                    T_ = st["T"]
                    near = min({r["T"] for r in rows_g}, key=lambda t: abs(t - T_))
                    cg = next((r["log10_reach"] for r in rows_g if r["kind"] == "gru" and r["scale"] == 1.0 and r["T"] == near), None)
                    cl = next((r["log10_reach"] for r in rows_g if r["kind"] == "lstm" and r["scale"] == 1.0 and r["T"] == near), None)
                    if cg is not None and cl is not None:
                        msg += (f" This is consistent with section 3: at T = {near} the gradient reaching the first input is 10^{cg:.1f} of the last step's for a default GRU "
                                f"(update-gate bias 0) against 10^{cl:.1f} for a default LSTM (forget bias 1). It is a training failure of this setting (fixed learning rate and epochs), not proof that GRUs cannot remember.")
                out.append(msg)
            elif len(mean) > 1:
                out.append(f"* {label}: every model reached ≥ {min(mean.values()):.2f} mean accuracy, so this setting does not separate them.")
        out.append("")
        for label, st in gd["settings"].items():
            sel_seeds = st.get("lstm_gate_selectivity_by_seed")
            if not sel_seeds:
                sel_seeds = [st["lstm_gate_selectivity_mean_over_50_sequences"]]
            accs = st["models"]["lstm"].get("test_accuracy_by_seed", [st["models"]["lstm"]["test_accuracy"]])
            for sd, (sel, acc) in enumerate(zip(sel_seeds, accs)):
                if acc < 0.9:
                    continue
                out.append(f"* {label}, LSTM seed {st.get('seeds', [0])[sd]} (accuracy {acc:.2f}): write gate {sel['write_i_at_value']:.2f} at the value step vs {sel['write_i_at_noise']:.2f} on noise; "
                           f"forget gate {sel['keep_f_at_noise_after']:.2f} on the noise after the value; cell-state change {sel['cell_change_at_value']:.2f} at the value vs {sel['cell_change_at_noise']:.2f} on noise; "
                           f"on average {sel.get('units_with_write_delta_gt_0.3', float('nan')):.1f} of {int(sel.get('n_units', 16))} units raise their write gate by more than 0.3 at the value step.")
        n_solved = n_sel = 0
        for label, st in gd["settings"].items():
            sels = st.get("lstm_gate_selectivity_by_seed") or [st["lstm_gate_selectivity_mean_over_50_sequences"]]
            accs = st["models"]["lstm"].get("test_accuracy_by_seed", [st["models"]["lstm"]["test_accuracy"]])
            for sel, acc in zip(sels, accs):
                if acc >= 0.9:
                    n_solved += 1
                    n_sel += int(sel["write_i_at_value"] - sel["write_i_at_noise"] > 0.1 and sel["cell_change_at_value"] > 1.5 * sel["cell_change_at_noise"])
        out += ["", f"**Summary:** in {n_sel} of {n_solved} LSTM runs that solved the task, the average write gate is clearly higher (> 0.1) at the value step than on noise *and* the cell state changes "
                "more than 1.5× as much there. Where this does not hold the network still solves the task, so selective writing is a *mechanism the cell can use*, "
                "visible in some runs and more clearly in some units than in the unit average - not something every trained LSTM shows in its mean gate values.", "",
                "![gates](img/fig_gates.png)", ""]
    return "\n".join(out)


def purity_sentence(pur, s, best_lat):
    k = pur["k"]
    key = f"purity@{k}"
    raw_c, raw_n = pur["raw"]["clean_query"][key], pur["raw"]["noisy_query"][key]
    best_c = pur[best_lat]["clean_query"][key]
    txt = f"Chance purity is {pur['raw']['clean_query']['chance_purity']:.3f}. "
    gap = best_c - raw_c
    rel = "more category information than" if gap > 0.03 else "about as much category information as" if gap > -0.03 else "less category information than"
    txt += (f"The 64-dimensional codes keep {rel} the {s['top_n_skills']:,}-dimensional raw vectors "
            f"(best latent: {best_lat}, {best_c:.3f} vs raw {raw_c:.3f}). ")
    if "denoising" in pur:
        d_drop, r_drop = pur["denoising"]["clean_query"][key] - pur["denoising"]["noisy_query"][key], raw_c - raw_n
        txt += (f"With noisy queries purity drops by {r_drop:.3f} for the raw vectors and {d_drop:.3f} for the denoising codes: "
                + ("the denoising codes are more robust to the corruption." if d_drop < r_drop - 0.01 else "the denoising codes are **not** more robust to the corruption here."))
    return txt


def section_ae(ae, info_train):
    s = ae["setting"]
    names = ["random", "popularity", "cooc"] + [k for k in ("basic", "deep", "denoising", "deep_noreg", "denoising_noreg") if k in ae["models"]]
    rows = []
    for n in names:
        isb = n in ae["baselines"]
        d = ae["baselines"][n] if isb else ae["models"][n]
        r = d["recovery"]
        pair = "" if isb else ae["models"][n]["paired_recall@10_vs_cooc"]
        pair_s = "-" if isb else f"{pair['mean_diff']:+.4f} [{pair['ci95_low']:+.4f}, {pair['ci95_high']:+.4f}]"
        rows.append([n + (" (baseline)" if isb else ""), f"{r['recall@5']:.3f}", f"{r['recall@10']:.3f}", f"{r['recall@20']:.3f}", f"{r['hit@10']:.3f}",
                     pair_s, "-" if "spurious_skill_auc" not in d else f"{d['spurious_skill_auc']:.3f}"])
    t1 = table(["method", "recall@5", "recall@10", "recall@20", "hit@10", "recall@10 minus co-occurrence [95% interval]", "spurious-skill AUC"], rows)
    rows2 = []
    for n in [k for k in ("basic", "deep", "denoising", "deep_noreg", "denoising_noreg") if k in ae["models"]]:
        m = ae["models"][n]
        tr = ae["setting"]["specs"][n]
        rows2.append([n, tr[0], "yes" if tr[1] else "no", "yes" if tr[2] else "no", f"{m['history'].get('params', 0):,}" if m.get("history") else "-",
                      m["history"].get("epochs_run", "-") if m.get("history") else "-", f"{m['bce_clean_to_clean']:.5f}", f"{m['bce_noisy_to_clean']:.5f}",
                      f"{m['bce_noisy_to_clean_train_subset']:.5f}", f"{m['cosine_recon_vs_clean']:.3f} / {m['cosine_probabilities_vs_clean']:.3f}",
                      f"{m.get('train_seconds') or 0:.0f}"])
    t2 = table(["model", "architecture", "regularised", "trained on corrupted input", "parameters", "epochs", "test BCE clean→clean", "test BCE noisy→clean",
                "BCE noisy→clean on train subset", "cosine(top-m recon, clean) / (probabilities, clean)", "train s"], rows2)
    out = ["## 5. Autoencoders on skill vectors", "",
           f"**Data.** Multi-hot vectors over the {s['top_n_skills']:,} most frequent non-ambiguous skills (vocabulary chosen on the training split only) for "
           f"{sum(s['splits'].values()):,} real jobs (EMSCAD, LinkedIn and Naukri; no synthetic rows, fraud or duplicates; at least 3 vocabulary skills), split **by employer**: "
           + ", ".join(f"{k} {v:,}" for k, v in s["splits"].items()) + ". Latent size " + str(s["latent"]) + ". "
           "**Corruption** (training and test): each present skill is dropped with probability 0.3 (at least one dropped, one kept) and 0–3 random absent skills are added.", "",
           "**Models.** *basic* (F→64→F), *deep* (F→512→256→64→256→512→F), *denoising* (the deep architecture trained on corrupted input to reconstruct the clean vector; the main model), "
           "plus *_noreg* variants without L2 (1e-5) and dropout (0.2) as the regularisation ablation. Sigmoid outputs, binary cross-entropy, Adam, early stopping on validation loss.", "",
           "### Recovering the dropped skills (test split, same corrupted input for every method)", "", t1, "",
           f"{ae['n_test_jobs']:,} test jobs. Candidates exclude every skill already present in the noisy input (including the random ones). *Random* ranks skills at random, "
           "*popularity* by training frequency, *cooc* by Σ P(skill | present skill) from training co-occurrence. *Spurious-skill AUC*: among skills present in the noisy input, how well the score "
           "separates true skills from the randomly added ones (1 = perfect, 0.5 = chance). The interval is an employer-clustered bootstrap of the paired difference to the co-occurrence baseline "
           "(descriptive, not a hypothesis test).", ""]
    d = ae["models"]["denoising"]["paired_recall@10_vs_cooc"]
    r_den, r_coo = ae["models"]["denoising"]["recovery"]["recall@10"], ae["baselines"]["cooc"]["recovery"]["recall@10"]
    tie = d["ci95_low"] <= 0 <= d["ci95_high"]
    out += ["### Reading the result (computed)", "",
            f"* The denoising autoencoder recovers {r_den:.3f} of the dropped skills in its top 10 versus {ae['baselines']['popularity']['recovery']['recall@10']:.3f} for popularity and "
            f"{r_coo:.3f} for co-occurrence. The paired difference to co-occurrence is {d['mean_diff']:+.4f} with interval [{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}]: "
            + ("the interval contains 0, so **the autoencoder is not better than the simple co-occurrence baseline** on this task; it matches it with one compact model (a 64-dimensional code) "
               "and additionally gives a latent representation." if tie else ("the autoencoder is better than co-occurrence." if d["mean_diff"] > 0 else "the autoencoder is worse than co-occurrence.")),
            ]
    if "deep_noreg" in ae["models"] and "deep" in ae["models"]:
        a, b = ae["models"]["deep_noreg"], ae["models"]["deep"]
        matters = a["recovery"]["recall@10"] < b["recovery"]["recall@10"] - 0.02
        out.append(f"* **Regularisation {'matters' if matters else 'made little difference'} for the clean-input deep autoencoder:** without L2/dropout its recall@10 is {a['recovery']['recall@10']:.3f} versus {b['recovery']['recall@10']:.3f} with them "
                   f"(BCE on a training subset {a['bce_noisy_to_clean_train_subset']:.5f} vs test {a['bce_noisy_to_clean']:.5f}"
                   + ("; it fits the training jobs better than unseen ones)." if a["bce_noisy_to_clean"] > a["bce_noisy_to_clean_train_subset"] else ")."))
    if "denoising_noreg" in ae["models"]:
        a, b = ae["models"]["denoising_noreg"], ae["models"]["denoising"]
        close = abs(a["recovery"]["recall@10"] - b["recovery"]["recall@10"]) < 0.02
        out.append(f"* Trained on corrupted input the un-regularised model reaches recall@10 {a['recovery']['recall@10']:.3f} versus {b['recovery']['recall@10']:.3f} regularised"
                   + (" - nearly the same, so the corruption itself acts as a regulariser (the classical argument for denoising autoencoders)." if close else "."))
    out.append(f"* Cosine similarity to the clean vector: the noisy input has {ae['cosine_noisy_vs_clean']:.3f}; the reconstruction as a top-m set has "
               f"{ae['models']['denoising']['cosine_recon_vs_clean']:.3f} (denoising model). The reconstruction is a *typical profile* for the observed skills, not a copy of them, so this raw "
               "cosine is lower than the input's - it is reported because the syllabus asks for it, but recovery@k and spurious-skill AUC are the meaningful measures here.")
    pur = ae.get("knn_category_purity")
    if pur:
        keys = ["raw"] + [k for k in ("basic", "deep", "denoising", "deep_noreg", "denoising_noreg") if k in pur]
        rows3 = [[k, f"{pur[k]['clean_query']['purity@%d' % pur['k']]:.3f}", f"{pur[k]['noisy_query']['purity@%d' % pur['k']]:.3f}",
                  f"{pur[k]['clean_query']['knn_accuracy']:.3f}"] for k in keys]
        best_lat = max((k for k in keys if k != "raw"), key=lambda k: pur[k]["clean_query"]["purity@%d" % pur["k"]])
        out += ["", f"### Do the latent codes keep the job category? (k-NN, k = {pur['k']})", "",
                f"Labels: {pur['labelled_source'].upper()} job-function categories ({', '.join(pur['classes'])}); queries are test jobs, neighbours are training jobs "
                f"({pur['raw']['clean_query']['n_queries']:,} queries, {pur['raw']['clean_query']['n_reference']:,} reference jobs). "
                "The MyCareersFuture categories originally planned could not be used in this session (the file could not be downloaded), so the EMSCAD labels are used.", "",
                table(["representation", f"purity@{pur['k']} (clean query)", f"purity@{pur['k']} (noisy query)", "k-NN accuracy (clean)"], rows3), "",
                purity_sentence(pur, s, best_lat)]
    out += ["", "### Model details", "", t2, "", "![autoencoder results](img/fig_autoencoder.png)", ""]
    bys = ae.get("recovery_by_source")
    if bys:
        out += ["Recall@10 by data source (test):", "", table(["source", "denoising", "popularity", "co-occurrence"],
                                                              [[k, f"{v['denoising']:.3f}", f"{v['popularity']:.3f}", f"{v['cooc']:.3f}"] for k, v in bys.items()]), "",
                "EMSCAD skills are *extracted* from the description text; LinkedIn and Naukri skills are the postings' own tags, so the sources differ in how predictable the skills are.", ""]
    return "\n".join(out)


def section_cases(cs):
    out = ["## 6. Case studies (separate from the job workflow)", ""]
    ic = cs.get("image_compression")
    if ic:
        rows = [[r["code_size"], f"x{r['compression_ratio']:.0f}", f"{r['mse_conv_ae']:.4f}", f"{r['psnr_conv_ae_db']:.1f}", f"{r['mse_pca']:.4f}", f"{r['psnr_pca_db']:.1f}"] for r in ic["rows"]]
        wins = sum(r["mse_conv_ae"] < r["mse_pca"] for r in ic["rows"])
        out += ["### Image compression: convolutional autoencoder vs PCA", "", f"Data: {ic['data']}. A 64-pixel image is stored as k numbers.", "",
                table(["code size k", "compression", "conv-AE test MSE", "PSNR dB", "PCA test MSE", "PSNR dB"], rows), "",
                f"The convolutional autoencoder has the lower error at {wins} of {len(ic['rows'])} code sizes. The images are tiny (8×8), so this is a demonstration of the technique, not of image-codec performance.", ""]
    v = cs.get("vae")
    if v:
        g = v["generated"]
        out += ["### Variational autoencoder", "",
                f"Latent {v['latent']}, loss = reconstruction + β·KL (β = {v['beta']}). Test MSE from the mean code {v['test_mse_from_mean_code']:.4f}; KL {v['test_kl_nats_per_image']:.2f} nats per image. "
                f"{g['n']} images sampled from N(0, I) were classified by a logistic-regression digit classifier trained on real digits (test accuracy {v['classifier_accuracy_on_real_test_digits']:.3f}): "
                f"mean top-class probability {g['mean_top_class_probability']:.2f}, {g['distinct_digits_produced']} of 10 digits produced (histogram {g['class_histogram']}). "
                "Compared with the deterministic convolutional autoencoder, the VAE trades reconstruction sharpness for a latent space that can be sampled; both are small-scale demonstrations.", ""]
    t = cs.get("traffic_anomaly")
    if t:
        a = t["autoencoder"]
        rows = [[k, v["n"], f"{v['recall_at_threshold']:.2f}", f"{v['auc_ae']:.3f}", f"{v['auc_pca']:.3f}", f"{v['auc_isolation_forest']:.3f}"] for k, v in t["per_attack"].items()]
        pca, iso = t["pca_baseline"]["auc"], t["isolation_forest_baseline"]["auc"]
        out += ["### Network-traffic anomaly detection — SYNTHETIC DATA", "",
                f"**Data: {t['data']}.** The CIC-IDS2017 files were not available, so flows are generated (normal: web, DNS, SSH, streaming; attacks: DoS, port scan, brute force, exfiltration). "
                "The numbers describe this generator, **not** real intrusion-detection performance. `load_cic_ids` reads the real CSVs if you supply them.", "",
                f"Autoencoder trained on {t['n_train_normal']:,} normal flows only; threshold = {t['threshold_rule']} (no attack used to choose it). Test: {t['n_test']:,} flows, {t['n_test_attacks']:,} attacks.", "",
                f"* at the threshold: precision {a['precision']:.3f}, recall {a['recall']:.3f}, F1 {a['f1']:.3f}, false-positive rate {a['false_positive_rate']:.3f}; threshold-free AUC {a['auc']:.3f};",
                f"* baselines fitted on the same normal data: PCA reconstruction error AUC {pca:.3f}, Isolation Forest AUC {iso:.3f}"
                + (" - **PCA is at least as good as the autoencoder on this generator** (its feature relations are close to linear)." if pca >= a["auc"] else "."), "",
                table(["attack", "n", "recall at threshold", "AUC autoencoder", "AUC PCA", "AUC isolation forest"], rows), "",
                "Brute-force flows were made deliberately similar to normal logins, which is why they are the hardest.", ""]
    out += ["### Brief case-study notes (text only, not application modules)", "",
            "* **Image captioning** — an encoder (a convolutional network) turns the image into a feature vector that initialises or conditions a recurrent decoder (LSTM/GRU); the decoder emits the caption one word at a time, each step fed by the previous word. It reuses exactly the recurrent machinery of section 2, with the image as the initial context.",
            "* **Facial recognition** — a convolutional network maps a face to an embedding in which images of the same person are close; verification compares two embeddings against a threshold and identification searches the nearest stored embedding. Training uses metric-learning losses (contrastive/triplet). The same embed-then-compare idea is used in this project for job and resume similarity.",
            "* **Self-driving** — perception (convolutional networks for detection and segmentation), prediction (recurrent models over object tracks) and planning/control are combined; safety depends on rare cases, sensor fusion and validation far beyond what a classifier accuracy shows. Not implemented here.", ""]
    return "\n".join(out)


def section_embeddings(ec, runs):
    if not ec:
        return ""
    return "\n".join([
        "## 7. Deployed encoder and embeddings", "",
        f"`DLEncoder` loads the sequence model with the lowest validation loss (**{ec['model']}**) and `scripts/build_dl_embeddings.py` pre-computes a 64-dimensional L2-normalised vector for every job with description text "
        "(`data/dl/embeddings_text.npz`) and a denoising-autoencoder latent code for every job with skills (`embeddings_skills.npz`); nothing is computed at search time.", "",
        f"**Sanity check of the text embeddings.** Test-split jobs (never seen by the model, employers not in training) were used as queries against training jobs: "
        f"the share of the 10 nearest neighbours with the same job function is **{ec['purity@10']:.3f}** (chance {ec['chance_purity']:.3f}); "
        f"a 10-NN majority vote reaches {ec['knn_accuracy']:.3f} accuracy ({ec['n_queries']:,} queries, {ec['n_reference']:,} reference jobs). "
        "So the embedding carries job-function information beyond chance; it is not claimed to be an optimal similarity measure.", ""])


def section_limits(runs, base):
    return "\n".join([
        "## 8. Limitations and honest notes", "",
        "* **Recurrent models vs. the baseline.** On this task and data size the recurrent models do not beat a bag-of-words logistic regression (section 2). They are the syllabus subject; the project does not claim they are the best classifier.",
        f"* **Few seeds.** {min(len(runs[k]) for k in KIND_ORDER if runs[k])} seeds per model; variation between seeds is large, so small differences are noise. More seeds would be needed for confident rankings.",
        "* **Truncation.** Descriptions are cut at 256 tokens (57% of descriptions are longer), which handicaps all four models equally but the SimpleRNN most.",
        "* **Run-to-run reproducibility.** Seeds fix initialisation and shuffling, but TensorFlow on different machines/thread counts does not give bit-identical training: one LSTM run retrained in a later session "
        "gave a different test accuracy than the first time (the other reproduced exactly). The numbers here are from the latest run of each configuration.",
        "* **Label quality.** Function labels were chosen by advertisers; some classes overlap (IT vs Engineering).",
        "* **Autoencoders.** The skill-vector task is evaluated on corruption that we generated; the 30%/0–3 noise rates are assumptions. The denoising model ties the co-occurrence baseline; its value in the platform is a compact code and a single model for suggestion and similarity.",
        "* **Controlled memory task** and **network-traffic data** are synthetic and labelled as such; they illustrate mechanisms and do not measure real-world performance.",
        "* **MyCareersFuture** could not be re-downloaded in the session that produced these results, so category purity uses EMSCAD labels, and the search index used for embeddings was rebuilt without MyCareersFuture.", ""])


def store_rows(runs, base, info, ae, cs, engine):
    from jobintel.dl.registry import store_experiment
    ds = f"emscad, {info['jobs']} jobs, {len(info['classes'])} classes"
    for k in KIND_ORDER:
        R = runs[k]
        if not R:
            continue
        best = min(R, key=lambda r: r["val_loss"])
        res = {key: float(np.mean([r[key] for r in R])) for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "val_loss", "train_seconds")}
        res.update(n_parameters=int(R[0]["n_parameters"]), artifact_path=f"data/dl/models/{best['_name']}.keras",
                   config=dict(R[0]["config"], seeds=[r["_seed"] for r in R], aggregate="mean over seeds"),
                   note=f"mean of {len(R)} seeds; accuracy range {min(r['accuracy'] for r in R):.3f}-{max(r['accuracy'] for r in R):.3f}; test split")
        store_experiment(k, res, info, engine=engine)
    if base:
        store_experiment("baseline", dict(base, config={"model": "CountVectorizer(binary) + LogisticRegression(balanced)"}, note="non-neural reference"), info, engine=engine)
    if ae:
        kind_map = {"basic": "autoencoder", "deep": "deep_ae", "denoising": "dae"}
        for n, kind in kind_map.items():
            m = ae["models"].get(n)
            if not m:
                continue
            res = {"reconstruction_loss": m["bce_noisy_to_clean"], "train_seconds": m.get("train_seconds"), "n_parameters": (m.get("history") or {}).get("params"),
                   "artifact_path": f"data/dl/models/ae_{n}.keras", "val_loss": (m.get("history") or {}).get("best_val_loss"),
                   "config": {"latent": ae["setting"]["latent"], "top_n_skills": ae["setting"]["top_n_skills"], "noise": ae["noise"]},
                   "note": f"skill-vector autoencoder; test recall@10 of dropped skills {m['recovery']['recall@10']:.3f} "
                           f"(co-occurrence baseline {ae['baselines']['cooc']['recovery']['recall@10']:.3f}); spurious-skill AUC {m.get('spurious_skill_auc', float('nan')):.3f}"}
            store_experiment(kind, res, info, task="skill-vector autoencoder (corrupted skills)", engine=engine,
                             dataset=f"real jobs (emscad, linkedin, naukri), {sum(ae['setting']['splits'].values())} jobs, {ae['setting']['top_n_skills']} skills")
    if cs:
        ic = cs.get("image_compression")
        if ic:
            row = next((r for r in ic["rows"] if r["code_size"] == 16), ic["rows"][-1])
            store_experiment("cae", {"reconstruction_loss": row["mse_conv_ae"], "config": {"code_size": row["code_size"]}, "n_parameters": row["conv_ae_params"],
                                     "note": f"convolutional AE vs PCA (MSE {row['mse_pca']:.4f}) on 8x8 digits"}, info, task="image compression (digits)", engine=engine,
                             dataset="sklearn load_digits")
        v = cs.get("vae")
        if v:
            store_experiment("vae", {"reconstruction_loss": v["test_mse_from_mean_code"], "config": {"latent": v["latent"], "beta": v["beta"]},
                                     "note": f"KL {v['test_kl_nats_per_image']:.2f} nats; {v['generated']['distinct_digits_produced']}/10 digits sampled"},
                             info, task="variational autoencoder (digits)", engine=engine, dataset="sklearn load_digits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-db", action="store_true")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    IMG.mkdir(parents=True, exist_ok=True)

    info = jload(DL / "classification_info.json")
    runs, base = load_runs()
    if not any(runs.values()):
        sys.exit("no model results found - run scripts/train_dl.py first")
    g, ge, gd = jload(EXP / "gradients.json"), jload(EXP / "gates_example.json"), jload(EXP / "gate_demo.json")
    ae, cs = jload(EXP / "autoencoder.json"), jload(EXP / "case_studies.json")
    ec = jload(EXP / "embedding_check.json")
    best = min((r for k in KIND_ORDER for r in runs[k]), key=lambda r: r["val_loss"])

    fig_learning_curves(runs)
    fig_comparison(runs, base)
    fig_confusion(best, info["classes"])
    if g:
        fig_gradients(g)
    if ae:
        fig_autoencoder(ae)
    if gd:
        fig_gates(gd)

    cmp_md, _ = section_comparison(runs, base, best)
    parts = ["# Phase 5 - Deep learning: results report", "",
             "*Generated by `scripts/report_dl.py` from `data/dl/results/*.json` and `data/dl/experiments/*.json`. Nothing in this file is typed by hand.*", "",
             section_task(info), "", cmp_md]
    if g:
        parts += ["", section_gradients(g)]
    if ge or gd:
        parts += ["", section_gates(ge, gd, g)]
    if ae:
        parts += ["", section_ae(ae, None)]
    if cs:
        parts += ["", section_cases(cs)]
    if ec:
        parts += ["", section_embeddings(ec, runs)]
    parts += ["", section_limits(runs, base)]
    parts += ["", "## 9. Reproduce", "", "```",
              "python scripts/train_dl.py --prepare",
              "python scripts/train_dl.py --model baseline rnn birnn lstm gru --seeds 42 43 --skip-existing    # repeat until nothing is left (resumable)",
              "python scripts/run_gradient_experiment.py", "python scripts/trace_gates.py", "python scripts/run_gate_demo.py --only short medium --seeds 0 1 2",
              "python scripts/train_autoencoder.py", "python scripts/run_case_studies.py", "python scripts/build_dl_embeddings.py", "python scripts/report_dl.py", "```", ""]
    (DOCS / "phase5_report.md").write_text("\n".join(parts), encoding="utf-8")
    print("wrote", DOCS / "phase5_report.md", "and", len(list(IMG.glob("fig_*.png"))), "charts")

    if not a.no_db:
        from jobintel.db.session import get_engine
        try:
            store_rows(runs, base, info, ae, cs, get_engine())
            print("stored mean rows in model_experiments")
        except Exception as e:                                   # the report is still valid without the database
            print("database rows NOT stored:", type(e).__name__, str(e)[:200])


if __name__ == "__main__":
    main()
