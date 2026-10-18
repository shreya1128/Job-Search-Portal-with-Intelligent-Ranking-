"""Parallel / distributed retrieval, simulated on one machine.

    CLIENT -> QUERY COORDINATOR -> SEARCH NODES (one per partition, each a separate process) -> RESULT MERGER -> RANKER -> CLIENT

 * SearchNode        owns one PartitionIndex; answers requests: local candidates, local scoring, local top-n.
 * Executors         HOW the coordinator reaches the nodes - all share  run(request) -> [node results]:
        SequentialExecutor   nodes are called one after another in the coordinator process   (the baseline)
        ThreadExecutor       one thread per node (CPU-bound NumPy: only partly parallel because of the GIL)
        ProcessExecutor      one persistent worker PROCESS per node, each loads its own partition, requests/responses over pipes
                             (true multi-core parallelism; works with 'spawn', i.e. on Windows)
 * ResultMerger      per-node top-n lists -> global top-n  (exactly the single-index ranking: the same global statistics and tie-breaking)
 * QueryCoordinator  scatter the query, gather and merge, report timings
 * DistributedEngine the normal SearchEngine whose first stage is delegated to the coordinator; passages / signals / explanations are
                     computed from the merged shortlist by the very same code as the single-process engine.
Scaling to several machines: replace the pipe by an RPC (HTTP/gRPC), put each partition on its own host, keep the same request/response
messages and merger; replicate partitions for fault tolerance and load balancing; the global statistics (df, cf, N) are a few MB and are
distributed to every node once.  See docs/PARALLEL_DISTRIBUTED.md."""
from __future__ import annotations

import multiprocessing as mp
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from jobintel.ir import models as M
from jobintel.ir.engine import SearchEngine, Stage
from jobintel.ir.partition import PartitionIndex, build_partitions, load_partition, save_partitions

SINGLE = ("baseline", "lm", "lm_bigram", "dfr")


# =========================================================================== search node
def node_search(part: PartitionIndex, req: Dict) -> Dict:
    t0 = time.perf_counter()
    qterms, model, topn, P = req["qterms"], req["model"], req["topn"], req["params"]
    cands = M.candidates(part, qterms)
    if req["allowed"] is not None:
        loc = part.local_lookup[req["allowed"]]
        cands = np.intersect1d(cands, loc[loc >= 0], assume_unique=True)
    out: Dict = {"n_candidates": int(len(cands)), "docs": np.zeros(0, dtype=np.int64)}
    if not len(cands):
        out["node_ms"] = (time.perf_counter() - t0) * 1000
        return out
    if model == "baseline":
        raw = M.keyword_baseline(part, qterms, cands)
        pos = M.top_positions(raw, topn)
        out["score"] = raw[pos]
    elif model == "lm":
        raw = M.unigram_lm(part, qterms, cands, smoothing=P["smoothing"], mu=P["mu"], lam=P["lam"])
        pos = M.top_positions(raw, topn)
        out["score"] = raw[pos]
    elif model == "lm_bigram":
        raw = M.bigram_lm(part, req["qseq"], cands, mu=P["mu"], mu_b=P["mu_b"], alpha=P["alpha"], cf_bigram=req["cf_bigram"])
        pos = M.top_positions(raw, topn)
        out["score"] = raw[pos]
    elif model == "dfr":
        raw = M.dfr(part, qterms, cands, model=P["dfr_model"], c=P["dfr_c"])
        pos = M.top_positions(raw, topn)
        out["score"] = raw[pos]
    else:                                                                                  # passage | hybrid
        lm_raw = M.unigram_lm(part, qterms, cands, smoothing=P["smoothing"], mu=P["mu"], lam=P["lam"])
        pos = M.top_positions(lm_raw, topn)
        dfr_raw = None
        if model == "hybrid":
            dfr_raw = M.dfr(part, qterms, cands, model=P["dfr_model"], c=P["dfr_c"])
            pos = np.unique(np.concatenate([pos, M.top_positions(dfr_raw, topn)]))
        out["lm"] = lm_raw[pos]
        if dfr_raw is not None:
            out["dfr"] = dfr_raw[pos]
    out["docs"] = part.global_idx[cands[pos]]
    out["node_ms"] = (time.perf_counter() - t0) * 1000
    return out


class SearchNode:
    def __init__(self, part: PartitionIndex):
        self.part = part

    def handle(self, req: Dict) -> Dict:
        op = req["op"]
        if op == "search":
            return node_search(self.part, req)
        if op == "bigram_counts":
            return {"counts": M.bigram_collection_counts(self.part, req["qseq"])}
        if op == "info":
            return {"part_id": self.part.part_id, "n_local": self.part.n_local, "memory_mb": self.part.memory_mb()}
        raise ValueError(f"unknown operation {op!r}")


# =========================================================================== executors
class SequentialExecutor:
    kind = "sequential"

    def __init__(self, parts: Sequence[PartitionIndex]):
        self.nodes = [SearchNode(p) for p in parts]
        self.n_nodes = len(self.nodes)

    def run(self, req: Dict) -> List[Dict]:
        return [n.handle(req) for n in self.nodes]

    def close(self) -> None:
        pass


class ThreadExecutor(SequentialExecutor):
    kind = "threads"

    def __init__(self, parts: Sequence[PartitionIndex]):
        super().__init__(parts)
        self.pool = ThreadPoolExecutor(max_workers=max(1, self.n_nodes))

    def run(self, req: Dict) -> List[Dict]:
        return list(self.pool.map(lambda n: n.handle(req), self.nodes))

    def close(self) -> None:
        self.pool.shutdown(wait=True)


def _worker_main(conn, path: str, node_id: int) -> None:           # module-level so it can be pickled by 'spawn' (Windows)
    node = SearchNode(load_partition(path))
    conn.send(("ready", node_id))
    while True:
        req = conn.recv()
        if req is None:
            break
        try:
            conn.send(node.handle(req))
        except Exception as exc:                                    # report instead of dying silently
            conn.send({"error": repr(exc)})
    conn.close()


class ProcessExecutor:
    kind = "processes"

    def __init__(self, partition_paths: Sequence[Path], start_method: Optional[str] = None):
        ctx = mp.get_context(start_method) if start_method else mp.get_context()
        self.conns, self.procs = [], []
        for i, path in enumerate(partition_paths):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker_main, args=(child, str(path), i), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)
        for c in self.conns:                                        # wait until every node has loaded its partition
            assert c.recv()[0] == "ready"
        self.n_nodes = len(self.conns)

    def run(self, req: Dict) -> List[Dict]:
        for c in self.conns:                                        # scatter: all nodes start working ...
            c.send(req)
        res = [c.recv() for c in self.conns]                        # ... gather
        for r in res:
            if "error" in r:
                raise RuntimeError(f"search node failed: {r['error']}")
        return res

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


# =========================================================================== merge + coordinate
class ResultMerger:
    @staticmethod
    def merge(model: str, results: List[Dict], topn: int) -> Stage:
        n_cand = int(sum(r["n_candidates"] for r in results))
        E = np.zeros(0, dtype=np.int64)
        docs = np.concatenate([r["docs"] for r in results]) if results else E
        if not len(docs):
            return Stage(n_cand, E, E)
        if model in SINGLE:
            score = np.concatenate([r["score"] for r in results if len(r["docs"])])
            order = np.lexsort((docs, -score))[:topn]
            ranked = docs[order]
            smap = {int(docs[i]): float(score[i]) for i in order}
            return Stage(n_cand, ranked, ranked, lm=smap if model in ("lm", "lm_bigram") else {}, dfr=smap if model == "dfr" else {}, score=smap)
        lm = np.concatenate([r["lm"] for r in results if len(r["docs"])])
        keep = np.lexsort((docs, -lm))[:topn]
        dfr = None
        if model == "hybrid":
            dfr = np.concatenate([r["dfr"] for r in results if len(r["docs"])])
            keep = np.concatenate([keep, np.lexsort((docs, -dfr))[:topn]])
        pool_idx = np.unique(docs[keep])                              # shortlist = union of the global top-n lists, in document order
        sel = np.isin(docs, pool_idx)
        lm_map = {int(d): float(v) for d, v in zip(docs[sel], lm[sel])}
        dfr_map = {int(d): float(v) for d, v in zip(docs[sel], dfr[sel])} if dfr is not None else {}
        return Stage(n_cand, E, pool_idx, lm=lm_map, dfr=dfr_map)


class QueryCoordinator:
    def __init__(self, executor):
        self.executor = executor

    def execute(self, model: str, qterms, qseq, allowed: Optional[np.ndarray], topn: int, params: Dict) -> Stage:
        t0 = time.perf_counter()
        cf_bigram = None
        if model == "lm_bigram":                                      # two rounds: global bigram collection counts first, then scoring
            shares = [r["counts"] for r in self.executor.run({"op": "bigram_counts", "qseq": list(qseq)})]
            cf_bigram = [float(sum(col)) for col in zip(*shares)]
        req = {"op": "search", "model": model, "qterms": list(qterms), "qseq": list(qseq), "allowed": allowed, "topn": topn, "params": params, "cf_bigram": cf_bigram}
        t1 = time.perf_counter()
        results = self.executor.run(req)
        t2 = time.perf_counter()
        stage = ResultMerger.merge(model, results, topn)
        t3 = time.perf_counter()
        node_ms = [r["node_ms"] for r in results]
        stage.extra = {"exec_ms": round((t2 - t1) * 1000, 2), "nodes_max_ms": round(max(node_ms), 2), "nodes_sum_ms": round(sum(node_ms), 2),
                       "merge_ms": round((t3 - t2) * 1000, 2), "coordination_ms": round((t2 - t1) * 1000 - max(node_ms) + (t1 - t0) * 1000, 2)}
        return stage


class DistributedEngine(SearchEngine):
    """SearchEngine whose retrieval stage runs on the partitions.  `index` stays in the coordinator for the vocabulary, the document
    metadata and PageRank (a real deployment would keep those in a lightweight metadata service)."""

    def __init__(self, index, docstore, coordinator: QueryCoordinator, **kw):
        super().__init__(index, docstore, **kw)
        self.coordinator = coordinator
        self._tmpdir = None

    def _stage(self, model, parsed, qterms, allowed, fetch_k, browse) -> Stage:
        if browse:                                                    # browse mode uses only metadata/PageRank - nothing to distribute
            return super()._stage(model, parsed, qterms, allowed, fetch_k, browse)
        cfg = self.config
        params = dict(smoothing=cfg.lm_smoothing, mu=cfg.mu, lam=cfg.lam, dfr_model=cfg.dfr_model, dfr_c=cfg.dfr_c, mu_b=cfg.bigram_mu_b, alpha=cfg.bigram_alpha)
        topn = cfg.pool if model in ("passage", "hybrid") else fetch_k
        return self.coordinator.execute(model, qterms, parsed.sequence(self.index), allowed, topn, params)

    @classmethod
    def build(cls, index, docstore, n_parts: int, executor: str = "sequential", strategy: str = "range", partition_dir: Optional[Path] = None,
              start_method: Optional[str] = None, **kw) -> "DistributedEngine":
        parts = build_partitions(index, n_parts, strategy)
        tmp = None
        if executor == "processes":
            if partition_dir is None:
                tmp = tempfile.TemporaryDirectory()
                partition_dir = Path(tmp.name)
            ex = ProcessExecutor(save_partitions(parts, partition_dir), start_method)
        elif executor == "threads":
            ex = ThreadExecutor(parts)
        elif executor == "sequential":
            ex = SequentialExecutor(parts)
        else:
            raise ValueError("executor must be 'sequential', 'threads' or 'processes'")
        eng = cls(index, docstore, QueryCoordinator(ex), **kw)
        eng._tmpdir = tmp
        eng.partitions = parts
        return eng

    def close(self) -> None:
        self.coordinator.executor.close()
        if self._tmpdir:
            self._tmpdir.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
