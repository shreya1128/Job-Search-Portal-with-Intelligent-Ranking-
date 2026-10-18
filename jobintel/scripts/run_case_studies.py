#!/usr/bin/env python3
"""Autoencoder case studies (separate from the job workflow): image compression, VAE, network-traffic anomaly detection (synthetic).
  python scripts/run_case_studies.py [--only compression vae traffic]      -> data/dl/experiments/case_studies.json"""
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
    ap.add_argument("--only", nargs="+", choices=["compression", "vae", "traffic"], default=["compression", "vae", "traffic"])
    ap.add_argument("--cic-csv", default=None, help="optional real CIC-IDS2017 CSV for the anomaly study")
    a = ap.parse_args()
    from jobintel.dl import case_studies as C

    path = ROOT / "data" / "dl" / "experiments" / "case_studies.json"
    out = json.load(open(path)) if path.exists() else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    if "compression" in a.only:
        t = time.time(); out["image_compression"] = C.image_compression_study(); print(f"compression {time.time() - t:.0f}s")
        for r in out["image_compression"]["rows"]:
            print(f"  code {r['code_size']:2d} (x{r['compression_ratio']:.0f}): conv-AE MSE {r['mse_conv_ae']:.4f} ({r['psnr_conv_ae_db']:.1f} dB)   PCA MSE {r['mse_pca']:.4f} ({r['psnr_pca_db']:.1f} dB)")
    if "vae" in a.only:
        t = time.time(); out["vae"] = C.vae_study(); print(f"vae {time.time() - t:.0f}s"); print("  ", json.dumps(out["vae"]))
    if "traffic" in a.only:
        data = None
        if a.cic_csv:
            import numpy as np
            X, y, cols = C.load_cic_ids(a.cic_csv, 200000)
            data = (np.abs(X), np.where(y, "attack", "web"))
        t = time.time(); out["traffic_anomaly"] = C.traffic_anomaly_study(data=data); print(f"traffic {time.time() - t:.0f}s")
        r = out["traffic_anomaly"]
        print("  AE", r["autoencoder"], "\n  PCA auc", r["pca_baseline"], "IF auc", r["isolation_forest_baseline"])
        for k, v in r["per_attack"].items():
            print(f"   {k:12s} n={v['n']} recall@thr {v['recall_at_threshold']:.2f}  AUC ae {v['auc_ae']:.3f} pca {v['auc_pca']:.3f} if {v['auc_isolation_forest']:.3f}")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
