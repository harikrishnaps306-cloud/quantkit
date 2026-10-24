# Benchmarks

All numbers below were measured locally on **one machine, one run session**.
They are not guarantees, say nothing about other hardware, and are not a
claim of being faster than any other library except where a like-for-like
comparison is shown and described.

* Hardware: Intel Core i9-14900HX (32 logical processors), 15.7 GB RAM
* OS: Windows 11 (10.0.26300), laptop power profile as found (not tuned)
* Toolchains: Rust 1.97.1 (release profile), Python 3.11.9, NumPy 2.4.6,
  SciPy 1.17.1
* Date: 2026-10-23

## The cost of the accurate far-tail method

The pricer and the implied-volatility solver were rewritten for tail
robustness (see `NUMERICS.md`, sections 1 and 4). Accuracy is bought with
speed, and it is not free:

| scalar benchmark (criterion, serial) | before | after |
|---|---|---|
| `bsm::price`, S=100 K=105 call | 17.8 ns | 32.4 ns |
| `implied_vol::solve`, same option | 118 ns | 883 ns |

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
| `batch::price`, 100k options | 9.96 ms (10.0 M/s) | 0.751 ms (133 M/s) |
| `batch::greeks`, 100k | 4.92 ms (20.3 M/s) | 1.24 ms (80.6 M/s) |
| `detailed::price`, 100k | 13.2 ms (7.6 M/s) | 1.43 ms (69.9 M/s) |
| `detailed::implied_vol`, 20k | 51.7 ms (0.39 M/s) | 3.49 ms (5.7 M/s) |
| `mc::european`, 1M paths | 12.3 ms (81 M paths/s) | 0.776 ms (1.29 G paths/s) |
| `mc::american_lsm`, 20k train + 20k eval, 50 steps | 37.3 ms | 32.1 ms |
| `dataset::generate`, 100k rows | 101 ms | 101 ms (not parallelised) |

For reference, the previous formulas on the same benchmark inputs ran
`batch::price` at 2.65 ms serial and `detailed::implied_vol` (20k) at 10.8 ms.

The Monte Carlo rows were re-measured after the overflow-safe rewrite (the
European estimator is about 30% slower serially than before it, 9.4 ms for 1M
paths; the other rows are from the earlier session).

Also in `benches/pricing.rs` (serial): binomial American put 5.7 us / 112 us /
430 us at 100 / 500 / 1000 steps.

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

## Refresh of 24 October 2026 (commit `2b5e59a` plus new benches)

Same machine and toolchain as above, criterion, serial unless noted, one run each, laptop not tuned; differences under about 10% are within noise. The tables above are from earlier code and were not re-measured except where listed here.

| benchmark | median |
|---|---|
| `bsm::price` | 40.8 ns (earlier table: 32.4 ns) |
| `implied_vol::solve` | 1.05 us (earlier: 883 ns) |
| binomial American put, 100 / 500 / 1000 steps | 6.9 / 128.6 / 483 us |
| **extreme log-domain tree** (S=K=100, sigma=100, T=1), European 100 / 500 / 1000 steps | 48.5 us / 1.33 ms / 4.72 ms |
| same, American 100 / 500 / 1000 steps | 136 us / 3.28 ms / 12.1 ms |
| risk, 1M returns: historical VaR / CVaR | 4.47 / 4.62 ms |
| parametric VaR / Sharpe / Sortino | 10.7 / 14.2 / 8.5 ms |
| max drawdown / annualized volatility | 3.85 / 11.1 ms |
| `norm_inv_cdf` p=0.3 / 1e-6 / 1e-300 | 30 / 51 / 632 ns |
| `batch::price`, 100k: serial / parallel | 10.6 ms / 1.62 ms |
| `batch::greeks`, 100k: serial / parallel | 5.07 / 1.91 ms |
| `detailed::price`, 100k: serial / parallel | 13.8 / 2.99 ms |
| `detailed::implied_vol`, 20k: serial / parallel | 56.4 / 7.78 ms |
| `mc::european`, 1M: serial / parallel | 13.2 / 1.60 ms |
| `mc::american_lsm` serial | 39.4 ms |
| `dataset::generate`, 100k | 104.6 ms |

The extreme American tree costs about 20x to 25x the ordinary American put at equal step count (136 vs 6.9 us, 3.28 ms vs 129 us, 12.1 ms vs 483 us; the cost is `ln`/`exp` per node), and only trees with `volatility * sqrt(T * steps) > 600` take it. Sharpe and Sortino now copy and rescale the input; I did not profile how much of their time that accounts for. The parallel runs are noisier than those in the earlier table (for example `batch::price` parallel was 0.75 ms then, 1.62 ms now); the parallel figures should not be compared across the two tables. The Python benchmark was not re-run.

## Reproducing

```bash
cargo bench --bench pricing
cargo bench --bench throughput
cargo bench --bench throughput --features parallel
cd python && maturin develop --release && cd ..
python benchmarks/python_bench.py
RAYON_NUM_THREADS=1 python benchmarks/python_bench.py
```
