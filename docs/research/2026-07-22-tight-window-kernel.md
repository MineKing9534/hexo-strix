# Tight VCF incremental-window kernel

The tight verdict-only leaf solver can maintain every structural length-`L`
window touched by a placement instead of rebuilding overlapping strips from
each stone. On HeXO's three winning axes, a placement belongs to exactly
`3 * L` windows: 18 at the live `L = 6` configuration. Make/unmake updates the
two player bitmasks for those windows; radius reachability remains a query-time
filter because stones outside a window can change it.

The implementation is leaf-only and opt-in through
`leaf_forcing_incremental_windows`. Root, PV, and wide solver paths retain the
legacy scanner. Search order, node accounting, and proof rules are unchanged.

## Controlled replay

`bench_leaf_forcing` replayed 20,000 leaves captured from the qhead checkpoint
match at depth cap 16 and a 1,000-node leaf budget. Both ordering modes had
zero verdict transitions and zero proof-depth changes between scanners.

| Ordering | Kernel | Wins | No | Budget | Calls/s | Mean | p95 | p99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy | Legacy scan | 2,724 | 14,842 | 2,434 | 874 | 1,059 us | 8,018 us | 10,691 us |
| Legacy | Incremental | 2,724 | 14,842 | 2,434 | 1,818 | 469 us | 3,437 us | 4,132 us |
| Proof-cost | Legacy scan | 2,748 | 14,850 | 2,402 | 873 | 1,061 us | 8,448 us | 10,534 us |
| Proof-cost | Incremental | 2,748 | 14,850 | 2,402 | 1,795 | 475 us | 3,470 us | 4,280 us |

That is 2.08x solver throughput under the production legacy ordering and
2.06x with proof-cost ordering.

## CUDA self-play timing caveat

A 32-game-per-arm live CUDA run was not a controlled throughput comparison:
thread scheduling generated different game workloads (1,040 versus 1,094
measured moves), while another SPRT shared the GPU. It reported 0.162 games/s
for legacy and 0.151 games/s for incremental, but the unequal work and changing
GPU contention make that pair unsuitable for a verdict. The same-leaf replay
above isolates the kernel and is the acceptance measurement; a future quiet-GPU
self-play run should compare moves/s on matched workloads.
