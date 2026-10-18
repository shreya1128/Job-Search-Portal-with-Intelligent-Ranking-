#!/usr/bin/env python3
"""Vanishing / exploding gradient experiment (syllabus: RNN training difficulties).

  python scripts/run_gradient_experiment.py            # full grid + clipping + short training comparison
  python scripts/run_gradient_experiment.py --quick    # tiny grid (for a smoke test)

Writes data/dl/experiments/gradients.json ; the numbers are measured, nothing is hard-coded.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-training", action="store_true", help="skip the short real-data training comparison")
    ap.add_argument("--out", default=str(ROOT / "data" / "dl" / "experiments" / "gradients.json"))
    a = ap.parse_args()

    import numpy as np
    from jobintel.dl import gradients as gr
    from jobintel.dl.tokenizer import DLTokenizer

    t0 = time.time()
    if a.quick:
        grid = gr.run_grid(scales=(0.5, 2.0), lengths=(10, 50), seeds=(0, 1), keep_profile_for=50)
    else:
        grid = gr.run_grid()
    print(f"gradient grid: {len(grid['rows'])} cells in {time.time() - t0:.0f}s")

    clip = gr.clipping_demo(T=50 if a.quick else 100, scale=2.0, clip=1.0)
    out = {"grid": grid, "clipping_weight_gradients": clip}

    if not a.no_training and not a.quick:
        z = np.load(ROOT / "data" / "dl" / "encoded.npz", allow_pickle=False)
        tok = DLTokenizer.load(ROOT / "data" / "dl" / "tokenizer.json")
        info = json.load(open(ROOT / "data" / "dl" / "classification_info.json"))
        t1 = time.time()
        out["training_with_vs_without_clipping"] = gr.training_with_and_without_clipping(
            z["X_train"], z["y_train"], tok.vocab_size, len(info["classes"]), scale=3.0, epochs=3)
        print(f"training comparison {time.time() - t1:.0f}s")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)

    # short readable table: log10 of (gradient at first step / gradient at last step)
    rows = grid["rows"]
    Ts = sorted({r["T"] for r in rows})
    print("\nlog10(grad at step 1 / grad at step T)   (T = %s)" % Ts)
    for kind in sorted({r["kind"] for r in rows}):
        for s in sorted({r["scale"] for r in rows}):
            vals = {r["T"]: r["log10_reach"] for r in rows if r["kind"] == kind and r["scale"] == s}
            print(f"  {kind:9s} scale {s:<4}", "  ".join(f"{vals.get(T, float('nan')):8.1f}" for T in Ts))
    print("\nclipping (T=%d, scale 2, clip 1):" % clip[0]["T"])
    for r in clip:
        print(f"  {r['kind']:9s} median weight-gradient norm {r['median_weight_grad_norm']:.3g} -> after clipping {r['after_clipping']:.3g}")
    for r in out.get("training_with_vs_without_clipping", []):
        print(f"  train {r['optimizer']:5s} clip={r['clipnorm']}: loss/epoch {[round(v, 3) for v in r['train_loss_by_epoch']]} "
              f"non-finite={r['any_nonfinite_loss']} train-acc {r['final_train_accuracy']:.3f}")


if __name__ == "__main__":
    main()
