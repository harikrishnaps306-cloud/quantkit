# Benchmarks

All numbers below were measured locally on **one machine, one run session**.
They are not guarantees, say nothing about other hardware, and are not a
claim of being faster than any other library except where a like-for-like
comparison is shown and described.

* Hardware: Intel Core i9-14900HX (32 logical processors), 15.7 GB RAM
* OS: Windows 11 (10.0.26300), laptop power profile as found (not tuned)
* Toolchains: Rust 1.97.1 (release profile), Python 3.11.9, NumPy 2.4.6,
  SciPy 1.17.1
* Date: 2026-10-23 (Python section and the 400,000-option sweep); 2026-10-24 for the
  Rust criterion tables, re-measured on commit `2b5e59a` plus the uncommitted
  extra benchmarks in `benches/pricing.rs`, serial then parallel, one run each
  with nothing else using the machine. Criterion's own `change:` lines in the
  output compare serial against the stored parallel run and are meaningless;
  they were ignored. An earlier attempt the same day overlapped with another
  benchmark process and was discarded.

## The cost of the accurate far-tail method

The pricer and the implied-volatility solver were rewritten for tail
robustness (see `NUMERICS.md`, sections 1 and 4). Accuracy is bought with
speed, and it is not free:

| scalar benchmark (criterion, serial) | before | after |
|---|---|---|
| `bsm::price`, S=100 K=105 call | 17.8 ns | 34.8 ns |
| `implied_vol::solve`, same option | 118 ns | 1.01 us |

The pricer evaluates the ordinary formula when an error estimate says it is
accurate to about `2e-14` and the stable quadrature method otherwise (about
150-400 ns per evaluation). Options that are far out of the money relative to
`sigma sqrt(T)` take the slow path, so the batch numbers below depend heavily
on how many such options the workload contains. The grid and random inputs used
here are spread widely, so they sit towards the slow end; a book concentrated
near the money would be faster. Loosening the switch point from 100 to 1000 in
`src/black.rs` made the 100k-option batch price run 2.4x faster in a trial at
the cost of up to `1e-13` instead of `2e-14` relative error; the default favours
accuracy.

## Rust (criterion, `cargo bench --bench throughput`, 3 s measurement, median)

Serial build vs `--features parallel` (rayon, 32 threads). Same inputs, same
code, results identical (checked by the test-suite, not by the benchmark).

| benchmark | serial | parallel |
|---|---|---|
| `batch::price`, 100k options | 10.25 ms (9.8 M/s) | 0.826 ms (121 M/s) |
| `batch::greeks`, 100k | 5.02 ms (19.9 M/s) | 1.27 ms (78.7 M/s) |
| `detailed::price`, 100k | 13.68 ms (7.3 M/s) | 1.49 ms (67.3 M/s) |
| `detailed::implied_vol`, 20k | 53.96 ms (0.37 M/s) | 3.85 ms (5.2 M/s) |
| `mc::european`, 1M paths | 12.38 ms (81 M paths/s) | 0.911 ms (1.10 G paths/s) |
| `mc::american_lsm`, 20k train + 20k eval, 50 steps | 37.38 ms | 32.79 ms |
| `dataset::generate`, 100k rows | 106.3 ms | 103.1 ms (not parallelised) |

For reference, the previous formulas on the same benchmark inputs ran
`batch::price` at 2.65 ms serial and `detailed::implied_vol` (20k) at 10.8 ms.

Between the 2026-10-23 and 2026-10-24 measurements the library changed only
in the audit commit (`2b5e59a`). The differences in the tables (up to about
+17% for parallel Monte Carlo, otherwise within about 5%) come from one run per
configuration on a laptop with default power settings and cannot be attributed
to the code changes without repeated runs, which were not done.

Also in `benches/pricing.rs` (serial): binomial American put 5.97 us / 118 us /
447 us at 100 / 500 / 1000 steps (were 5.7 / 112 / 430 us; about +4-5%, with the
same caveat).

### Extreme binomial trees (log-domain rollback)

`binomial::price` takes the log-domain path when the node-to-node jump times the
step count exceeds 600 (see `NUMERICS.md`). Case: S = K = 100, volatility 100,
expiry 1, call, serial. Ordinary trees do not use this path.

| steps | European | American |
|---|---|---|
| 100 | 45.2 us | 116.0 us |
| 500 | 1.084 ms | 2.752 ms |
| 1000 | 4.379 ms | 10.77 ms |

For comparison, an ordinary American put tree at 1000 steps takes 447 us, so the
extreme path is about 24x slower for American exercise at the same step count
(different payoff and parameters, so this is a rough scale, not a like-for-like
regression). The cause was not profiled. There is no earlier measurement of this
path; before the fix it returned wrong values (zero for a call worth about 100).

### Risk functions, 1,000,000 returns (serial, median)

| function | time |
|---|---|
| `historical_var` (99%) | 3.01 ms |
| `historical_cvar` (99%) | 3.27 ms |
| `parametric_var` (99%) | 10.21 ms |
| `sharpe_ratio` | 13.16 ms |
| `sortino_ratio` | 8.00 ms |
| `max_drawdown` | 3.08 ms |
| `annualized_volatility` | 5.81 ms |

The audit commit made these scale-safe (normalised arithmetic); this is the
first measurement of them, so there is no before/after comparison and no claim
about how much the scaling cost.

### Normal quantile (`dist::norm_inv_cdf`, serial)

| input | time |
|---|---|
| p = 0.3 | 12.7 ns |
| p = 1e-6 | 46.8 ns |
| p = 1e-300 (log-tail refinement) | 611 ns |

Notes: LSM parallelises only the evaluation phase, so it gains little. Parallel
speed-ups depend on the number of cores and will be smaller on smaller machines.

A random sweep of 400,000 self-priced options over the whole supported domain
(`implied_vol::solve_detailed`, release, serial; run once with 16-node
quadrature, before the final change to 12 nodes) took 1.9 us per solve on
average with a mean of 5.5 iterations, a maximum of 28, and no bisection steps
needed; that average includes the in-the-money rows that are rejected before any
iteration.

## Python (`benchmarks/python_bench.py`, median of 15 runs, 1,000,000 rows)

Baseline: a NumPy + `scipy.special.ndtr` implementation of the same
Black-Scholes-Merton call formula (written for this benchmark; results agree
with quantkit to `8.5e-14` over all rows, with the NumPy version being the less
accurate in the tails).

| call | all threads | `RAYON_NUM_THREADS=1` |
|---|---|---|
| `quantkit.bs_price_batch` | 29.0 ms (34.5 M rows/s) | 152.5 ms (6.6 M rows/s) |
| NumPy + SciPy baseline | 75.3 ms (13.3 M rows/s) | 75.7 ms (13.2 M rows/s) |
| `quantkit.bs_price_detailed` (status + messages) | 38.1 ms | 162.3 ms |
| `quantkit.bs_greeks_batch` | 40.5 ms | 122.3 ms |
| `quantkit.implied_vol_batch`, 200k rows | 29.3 ms | 343.0 ms |
| `quantkit.implied_vol_detailed`, 200k rows | 32.9 ms | 340.4 ms |
| `bs_price_batch`, 10 rows (per call) | 2.8 us | 3.1 us |
| NumPy + SciPy, 10 rows (per call) | 7.0 us | 5.8 us |

How to read this fairly:

* **Single-threaded, quantkit's price batch is about 2x slower than the
  NumPy/SciPy baseline on this input** (152 vs 76 ms). It was roughly on par
  (76 vs 77 ms) before the stable far-tail method, which the inputs here
  (strikes 60-140 vs spot 50-150, volatilities 0.1-0.8, expiries 0.05-3 years)
  exercise often. The multi-thread result (about 2.6x faster than NumPy) comes
  from using 32 cores; NumPy/SciPy above is single-threaded. Neither number says
  anything about accuracy, where quantkit is the better of the two in the tails.
* The share of the single-thread time spent converting NumPy arrays to Rust
  vectors and back was not measured separately. A first version that built a
  struct per row was clearly slower than the columnar path now used.
* Not measured: comparisons with `py_vollib_vectorized`, `pyvolr`, QuantLib,
  or any implied-volatility competitor. Their speed claims are theirs; none
  were reproduced here.

## Reproducing

```bash
cargo bench --bench pricing
cargo bench --bench throughput
cargo bench --bench throughput --features parallel
cd python && maturin develop --release && cd ..
python benchmarks/python_bench.py
RAYON_NUM_THREADS=1 python benchmarks/python_bench.py
```
