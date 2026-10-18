# Parallel retrieval, cluster-based retrieval, distributed simulation, MapReduce (Phase 3)

## 1. Partitioned (parallel) retrieval
```
CLIENT -> QUERY COORDINATOR --scatter--> SEARCH NODE 1 (partition 1)  \
                             --scatter--> SEARCH NODE 2 (partition 2)   >-- local candidates + local scoring + local top-n
                             --scatter--> SEARCH NODE P (partition P)  /
          <-- RESULT MERGER (global top-n) <--gather--  -> RANKER (passages, signals, explanation) -> CLIENT
```
* **Partitions** (`ir/partition.py`): `range` (documents 0..N/P-1, N/P..2N/P-1, ... as in the brief) or `roundrobin`. A partition holds local postings, lengths and token sequences but **global** collection statistics (cf, df, N, average length), so it scores every document exactly as the whole index would.
* **Nodes and executors** (`ir/distributed.py`): `SequentialExecutor` (one node after another), `ThreadExecutor`, `ProcessExecutor` (one persistent worker process per partition, each loads its own partition file, request/response over pipes; works with `spawn`, i.e. on Windows).
* **Merger**: per-node top-n lists -> global top-n. Hybrid: the shortlist is the union of the global top-n by LM and by DFR, exactly as in the single-process engine. The bigram model needs a two-round protocol (nodes first return their share of the global bigram collection counts, the coordinator sums them, then scoring).
* **Exactness** (verified, not assumed): a document's score depends only on the document and global statistics; ties are broken by document order everywhere; the global top-n is always contained in the union of the local top-n. `scripts/verify_phase3.py` compares 376 searches on the real 60,808-document index (4 configurations incl. real worker processes, 6 models, filters, cluster routing): **0 differences** in job ids, scores and candidate counts.

### Benchmark (`scripts/benchmark_parallel.py`, results in `docs/parallel_benchmark.md`)
Measured here on **1 CPU core** - so true parallel speedup cannot appear and the measured speedup of the parallel executors is ~1x (0.64x-1.22x, i.e. noise plus small overhead). What the run does show, from per-node times: the **ideal speedup on P free cores** (an estimate, not a measurement) is 1.6x for P=2 and 2.7x for P=4 at 60,808 documents, and 3.5x-3.8x for P=4 at 243,232 documents (collection repeated 4x), because fixed costs matter less as the collection grows. At 60k documents a query already needs only 3-4 ms, so parallelism pays off only for much larger collections. **Run the script on your machine** to get real multi-core numbers (it prints speedup, efficiency = speedup/P, per-node and merge times).

## 2. Cluster-based retrieval (`ir/cluster.py`)
Jobs belong to 12 title-keyword clusters (Software/IT, Data/AI, Finance, Healthcare plus education, engineering/manufacturing, sales/marketing, HR/admin, retail/hospitality, logistics, legal/public, other). A query is routed by a naive-Bayes style classifier over cluster language models; only the clusters holding >= 90% of the posterior mass (max 3) are searched. Measured on 18 queries: 18% of the documents searched on average, candidates -67%, top-10 overlap with the full search **0.88**. It can also fail: `devops engineer docker kubernetes aws` was routed to Data/AI and kept only 2 of the full search's top 10 (overlap 0.20). No latency gain was measurable at this size. Clusters are heuristics, not labels, so routing is **opt-in** (`cluster_routing=True` / `--route`); Phase 5's neural classifier is the intended better router.

## 3. Distributed architecture (simulation)
`Client -> Query Coordinator -> Search Nodes (partitions) -> Result Merger -> Ranker -> Client` is implemented with local processes. To run it on several machines: (1) put each partition file on its own host and start a node service there; (2) replace the pipe by an RPC (HTTP/gRPC) carrying the same request/response dictionaries; (3) distribute the global statistics (a few MB) to every node once; (4) replicate partitions for fault tolerance and load balancing, with the coordinator sending a request to any healthy replica and re-trying on timeout; (5) shard by document id so adding a node means moving partitions, not re-indexing; (6) keep the coordinator stateless and put a cache of popular queries in front of it. The merge logic and the exactness argument do not change.

## 4. MapReduce (`mapreduce/`)
A real map -> combine -> shuffle (stable crc32 hash, safe under `spawn`) -> reduce engine on worker processes. Jobs: **market analysis** (skill frequency overall and by role cluster, city, experience band and month; skill pairs with lift; distributions; written to `data/analytics/market.json` for the Phase-7 dashboard) and **inverted-index construction**. Validation (`scripts/run_mapreduce.py`): 12,414 (skill, type) counts identical to SQL `GROUP BY`; three skill-pair counts and one cluster count identical to SQL; the postings of the first 10,000 documents built by MapReduce identical to the real index. The combiner removes 86% of the 3.4 M emitted pairs before the shuffle. With 2 workers on this 1-core machine the run is slower (0.57x) - again a property of the machine, not the method.

## 5. Windows note
Any script that starts worker processes must be importable without side effects (`if __name__ == "__main__":`); `spawn` re-imports the script in every worker. All provided scripts follow this.
