"""Sequential vs parallel retrieval benchmark (first stage = candidate retrieval + scoring + merge, no passages/DB).

For every (collection size, partition count P, executor, model) it records the WALL-CLOCK time of a query, the per-node compute times and
the merge time, then derives
   speedup     = T(sequential over the same P partitions) / T(executor)
   efficiency  = speedup / P
   ideal estimate = (sum of node times + merge) / (max node time + merge): the speedup P free cores would give if communication cost nothing.
The estimate is derived from measured node times but is NOT a measurement of parallel wall-clock - on a machine with fewer cores than P
the measured speedup can be below 1 (the parallel version then only adds overhead), which is itself a legitimate finding."""
from __future__ import annotations

import multiprocessing as mp
import platform
import statistics as st
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from jobintel.ir.distributed import ProcessExecutor, QueryCoordinator, SequentialExecutor, ThreadExecutor
from jobintel.ir.partition import build_partitions, replicate, save_partitions
from jobintel.ir.ranker import RankerConfig

BENCH_QUERIES = ["python machine learning sql", "registered nurse", "accountant tax", "java spring boot microservices", "data analyst sql excel",
                 "customer service representative", "devops docker kubernetes aws", "project manager agile scrum"]


def machine_info() -> Dict:
    import numpy
    return {"cpu_cores": mp.cpu_count(), "platform": platform.platform(), "python": platform.python_version(), "numpy": numpy.__version__}


def _params(cfg: RankerConfig) -> Dict:
    return dict(smoothing=cfg.lm_smoothing, mu=cfg.mu, lam=cfg.lam, dfr_model=cfg.dfr_model, dfr_c=cfg.dfr_c, mu_b=cfg.bigram_mu_b, alpha=cfg.bigram_alpha)


def run_benchmark(ix, parsed_queries: Sequence, sizes: Sequence[int] = (1,), parts: Sequence[int] = (1, 2, 4), executors: Sequence[str] = ("sequential", "threads", "processes"),
                  models: Sequence[str] = ("lm", "hybrid"), reps: int = 3, start_method=None, progress=print) -> List[Dict]:
    """parsed_queries: [(qterms, qseq)] with term ids of `ix`."""
    cfg = RankerConfig()
    P_ = _params(cfg)
    rows: List[Dict] = []
    for m in sizes:
        big = ix if m == 1 else replicate(ix, m)
        n_docs = len(big.doc_len)
        for P in parts:
            plist = build_partitions(big, P)
            for kind in executors:
                tmp = None
                t_start = time.perf_counter()
                if kind == "processes":
                    tmp = tempfile.TemporaryDirectory()
                    ex = ProcessExecutor(save_partitions(plist, Path(tmp.name)), start_method)
                elif kind == "threads":
                    ex = ThreadExecutor(plist)
                else:
                    ex = SequentialExecutor(plist)
                startup = time.perf_counter() - t_start
                coord = QueryCoordinator(ex)
                for model in models:
                    topn = cfg.pool if model in ("passage", "hybrid") else 10
                    for qt, qs in parsed_queries[:1]:                               # warm-up (caches, worker start)
                        coord.execute(model, qt, qs, None, topn, P_)
                    wall, nmax, nsum, merge, cand = [], [], [], [], []
                    for qt, qs in parsed_queries:
                        for _ in range(reps):
                            t0 = time.perf_counter()
                            stage = coord.execute(model, qt, qs, None, topn, P_)
                            wall.append((time.perf_counter() - t0) * 1000)
                            nmax.append(stage.extra["nodes_max_ms"]); nsum.append(stage.extra["nodes_sum_ms"]); merge.append(stage.extra["merge_ms"]); cand.append(stage.n_candidates)
                    wall.sort()
                    rows.append(dict(size=m, docs=n_docs, parts=P, executor=kind, model=model, median_ms=st.median(wall), p95_ms=wall[int(0.95 * (len(wall) - 1))],
                                     nodes_max_ms=st.mean(nmax), nodes_sum_ms=st.mean(nsum), merge_ms=st.mean(merge), candidates=st.mean(cand), startup_s=startup))
                    progress(f"  size x{m} ({n_docs:,} docs) P={P} {kind:10s} {model:7s} median {st.median(wall):7.1f} ms")
                ex.close()
                if tmp:
                    tmp.cleanup()
    # ---- derived columns -------------------------------------------------------------------------------------------------------------
    base = {(r["size"], r["parts"], r["model"]): r for r in rows if r["executor"] == "sequential"}
    single = {(r["size"], r["model"]): r for r in rows if r["executor"] == "sequential" and r["parts"] == 1}
    for r in rows:
        s = base[(r["size"], r["parts"], r["model"])]
        r["speedup_vs_sequential"] = s["median_ms"] / r["median_ms"]
        r["efficiency"] = r["speedup_vs_sequential"] / r["parts"]
        r["ideal_speedup_estimate"] = (s["nodes_sum_ms"] + s["merge_ms"]) / (s["nodes_max_ms"] + s["merge_ms"])
        if (r["size"], r["model"]) in single:
            r["speedup_vs_single_index"] = single[(r["size"], r["model"])]["median_ms"] / r["median_ms"]
    return rows


def to_markdown(rows: List[Dict], info: Dict) -> str:
    out = [f"Machine: {info['cpu_cores']} CPU core(s), {info['platform']}, Python {info['python']}, NumPy {info['numpy']}\n"]
    for size in sorted({r["size"] for r in rows}):
        for model in sorted({r["model"] for r in rows}):
            sel = [r for r in rows if r["size"] == size and r["model"] == model]
            out.append(f"\n**Collection x{size} = {sel[0]['docs']:,} documents, model `{model}`** (median ms per query; mean candidates scored {sel[0]['candidates']:,.0f})\n")
            out.append("| partitions P | executor | median ms | p95 ms | speedup vs sequential | efficiency | ideal speedup (estimate) | max node ms | sum node ms | merge ms |\n|---|---|---|---|---|---|---|---|---|---|")
            for r in sorted(sel, key=lambda r: (r["parts"], ["sequential", "threads", "processes"].index(r["executor"]))):
                out.append(f"| {r['parts']} | {r['executor']} | {r['median_ms']:.1f} | {r['p95_ms']:.1f} | {r['speedup_vs_sequential']:.2f}x | {r['efficiency']:.2f} | {r['ideal_speedup_estimate']:.2f}x | "
                           f"{r['nodes_max_ms']:.1f} | {r['nodes_sum_ms']:.1f} | {r['merge_ms']:.1f} |")
    return "\n".join(out)
