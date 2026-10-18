# Sequential vs parallel retrieval benchmark

Machine: 1 CPU core(s), Linux-6.18.44-fc-v64-x86_64-with-glibc2.39, Python 3.12.3, NumPy 2.4.4


**Collection x1 = 60,808 documents, model `hybrid`** (median ms per query; mean candidates scored 17,807)

| partitions P | executor | median ms | p95 ms | speedup vs sequential | efficiency | ideal speedup (estimate) | max node ms | sum node ms | merge ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sequential | 3.5 | 8.3 | 1.00x | 1.00 | 1.00x | 4.1 | 4.1 | 0.2 |
| 1 | threads | 3.4 | 9.0 | 1.03x | 1.03 | 1.00x | 4.0 | 4.0 | 0.2 |
| 1 | processes | 4.3 | 9.6 | 0.80x | 0.80 | 1.00x | 4.5 | 4.5 | 0.3 |
| 2 | sequential | 4.5 | 8.6 | 1.00x | 0.50 | 1.64x | 2.7 | 4.6 | 0.3 |
| 2 | threads | 3.9 | 9.2 | 1.16x | 0.58 | 1.64x | 3.6 | 5.5 | 0.3 |
| 2 | processes | 6.2 | 18.2 | 0.73x | 0.37 | 1.64x | 6.3 | 10.8 | 0.6 |
| 4 | sequential | 4.0 | 8.9 | 1.00x | 0.25 | 2.65x | 1.5 | 4.5 | 0.3 |
| 4 | threads | 4.0 | 8.9 | 1.01x | 0.25 | 2.65x | 3.5 | 7.7 | 0.3 |
| 4 | processes | 5.7 | 10.4 | 0.71x | 0.18 | 2.65x | 2.9 | 6.7 | 0.3 |

**Collection x1 = 60,808 documents, model `lm`** (median ms per query; mean candidates scored 17,807)

| partitions P | executor | median ms | p95 ms | speedup vs sequential | efficiency | ideal speedup (estimate) | max node ms | sum node ms | merge ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sequential | 2.6 | 7.1 | 1.00x | 1.00 | 1.00x | 3.2 | 3.2 | 0.0 |
| 1 | threads | 3.2 | 7.3 | 0.81x | 0.81 | 1.00x | 3.7 | 3.7 | 0.0 |
| 1 | processes | 3.0 | 7.5 | 0.85x | 0.85 | 1.00x | 3.6 | 3.6 | 0.0 |
| 2 | sequential | 2.6 | 7.9 | 1.00x | 0.50 | 1.63x | 2.1 | 3.5 | 0.0 |
| 2 | threads | 2.7 | 7.1 | 0.95x | 0.48 | 1.63x | 2.5 | 3.9 | 0.0 |
| 2 | processes | 4.1 | 10.3 | 0.64x | 0.32 | 1.63x | 3.6 | 5.7 | 0.0 |
| 4 | sequential | 2.5 | 7.2 | 1.00x | 0.25 | 2.75x | 1.2 | 3.3 | 0.0 |
| 4 | threads | 2.7 | 6.9 | 0.94x | 0.23 | 2.75x | 2.5 | 5.1 | 0.0 |
| 4 | processes | 3.4 | 7.8 | 0.74x | 0.19 | 2.75x | 1.8 | 4.3 | 0.0 |

**Collection x4 = 243,232 documents, model `hybrid`** (median ms per query; mean candidates scored 71,228)

| partitions P | executor | median ms | p95 ms | speedup vs sequential | efficiency | ideal speedup (estimate) | max node ms | sum node ms | merge ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sequential | 14.0 | 41.5 | 1.00x | 1.00 | 1.00x | 18.2 | 18.2 | 0.3 |
| 1 | threads | 14.5 | 39.5 | 0.97x | 0.97 | 1.00x | 18.0 | 18.0 | 0.3 |
| 1 | processes | 14.3 | 39.2 | 0.98x | 0.98 | 1.00x | 17.6 | 17.6 | 0.3 |
| 2 | sequential | 11.8 | 34.6 | 1.00x | 0.50 | 1.94x | 7.8 | 15.3 | 0.2 |
| 2 | threads | 13.4 | 35.2 | 0.88x | 0.44 | 1.94x | 14.9 | 28.3 | 0.3 |
| 2 | processes | 13.3 | 35.1 | 0.88x | 0.44 | 1.94x | 12.8 | 24.6 | 0.3 |
| 4 | sequential | 14.4 | 33.9 | 1.00x | 0.25 | 3.50x | 4.4 | 16.2 | 0.3 |
| 4 | threads | 13.3 | 34.6 | 1.08x | 0.27 | 3.50x | 14.4 | 44.5 | 0.3 |
| 4 | processes | 14.4 | 35.4 | 1.00x | 0.25 | 3.50x | 13.9 | 45.9 | 0.3 |

**Collection x4 = 243,232 documents, model `lm`** (median ms per query; mean candidates scored 71,228)

| partitions P | executor | median ms | p95 ms | speedup vs sequential | efficiency | ideal speedup (estimate) | max node ms | sum node ms | merge ms |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sequential | 13.7 | 35.0 | 1.00x | 1.00 | 1.00x | 16.6 | 16.6 | 0.0 |
| 1 | threads | 11.4 | 33.9 | 1.20x | 1.20 | 1.00x | 15.9 | 15.9 | 0.0 |
| 1 | processes | 11.3 | 32.9 | 1.22x | 1.22 | 1.00x | 14.4 | 14.4 | 0.0 |
| 2 | sequential | 9.4 | 26.3 | 1.00x | 0.50 | 1.95x | 6.1 | 11.9 | 0.0 |
| 2 | threads | 10.3 | 28.9 | 0.92x | 0.46 | 1.95x | 11.6 | 21.6 | 0.0 |
| 2 | processes | 10.0 | 29.7 | 0.94x | 0.47 | 1.95x | 9.9 | 19.0 | 0.0 |
| 4 | sequential | 9.0 | 24.1 | 1.00x | 0.25 | 3.80x | 3.0 | 11.3 | 0.0 |
| 4 | threads | 9.8 | 32.7 | 0.92x | 0.23 | 3.80x | 12.5 | 39.8 | 0.0 |
| 4 | processes | 10.8 | 29.1 | 0.83x | 0.21 | 3.80x | 10.7 | 30.8 | 0.0 |