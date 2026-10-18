"""A small MapReduce engine (Python, multiprocessing) - no Hadoop needed.

    records --split--> chunks --MAP (worker processes)--> (key, value) pairs --COMBINE (per chunk)--> --SHUFFLE (hash(key) -> reducer)-->
            --REDUCE (worker processes)--> {key: result}

 * mapper(record)            yields (key, value) pairs
 * combiner(key, [values])   optional local pre-aggregation inside each map task (cuts the data that has to be shuffled)
 * reducer(key, [values])    produces the final value for one key
Keys are routed to reducers with a STABLE hash (crc32): Python's built-in hash() is salted per process and would send the same key to
different reducers when workers are started with 'spawn' (Windows).  mapper / reducer / combiner must be module-level functions."""
from __future__ import annotations

import multiprocessing as mp
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple


def stable_hash(key: Hashable) -> int:
    return zlib.crc32(repr(key).encode("utf-8"))


def sum_reducer(key, values):
    return sum(values)


def _map_chunk(args):
    chunk, mapper, combiner, n_red = args
    buckets: List[Dict[Hashable, list]] = [defaultdict(list) for _ in range(n_red)]
    emitted = 0
    for rec in chunk:
        for k, v in mapper(rec):
            emitted += 1
            buckets[stable_hash(k) % n_red][k].append(v)
    out = [{k: ([combiner(k, vs)] if combiner else vs) for k, vs in b.items()} for b in buckets]
    return out, emitted, sum(len(vs) for b in out for vs in b.values())


def _reduce_part(args):
    part, reducer = args
    return {k: reducer(k, vs) for k, vs in part.items()}


@dataclass
class MRResult:
    output: Dict[Hashable, Any]
    stats: Dict[str, float] = field(default_factory=dict)


def run_mapreduce(records: Sequence, mapper: Callable, reducer: Callable, *, n_workers: int = 1, chunk_size: int = 2000, combiner: Optional[Callable] = None,
                  n_reducers: Optional[int] = None, start_method: Optional[str] = None) -> MRResult:
    n_red = n_reducers or max(1, n_workers)
    chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
    tasks = [(c, mapper, combiner, n_red) for c in chunks]
    pool = mp.get_context(start_method).Pool(n_workers) if n_workers > 1 else None
    try:
        t0 = time.perf_counter()
        mapped = pool.map(_map_chunk, tasks) if pool else [_map_chunk(t) for t in tasks]
        t1 = time.perf_counter()
        inputs: List[Dict[Hashable, list]] = [defaultdict(list) for _ in range(n_red)]          # SHUFFLE: group by key on the reducer side
        emitted = after = 0
        for out, e, a in mapped:
            emitted += e
            after += a
            for r, b in enumerate(out):
                for k, vs in b.items():
                    inputs[r][k].extend(vs)
        t2 = time.perf_counter()
        parts = pool.map(_reduce_part, [(inp, reducer) for inp in inputs]) if pool else [_reduce_part((inp, reducer)) for inp in inputs]
        t3 = time.perf_counter()
    finally:
        if pool:
            pool.close()
            pool.join()
    output: Dict[Hashable, Any] = {}
    for p in parts:
        output.update(p)
    return MRResult(output, dict(records=len(records), chunks=len(chunks), workers=n_workers, reducers=n_red, pairs_emitted=emitted, pairs_after_combine=after,
                                 keys=len(output), map_s=round(t1 - t0, 3), shuffle_s=round(t2 - t1, 3), reduce_s=round(t3 - t2, 3), total_s=round(t3 - t0, 3)))
