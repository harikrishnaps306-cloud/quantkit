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

## Rust (criterion, `cargo bench --bench throughput`, 3 s measurement, median)

Serial build vs `--features parallel` (rayon, 32 threads). Same inputs, same
code, results identical (checked by the test-suite, not by the benchmark).

| benchmark | serial | parallel |
|---|---|---|
| `batch::price`, 100k options | 2.65 ms (37.8 M/s) | 0.316 ms (316 M/s) |
| `batch::greeks`, 100k | 5.00 ms (20.0 M/s) | 1.22 ms (82 M/s) |
| `detailed::price`, 100k | 3.94 ms (25.4 M/s) | 1.36 ms (73.6 M/s) |
| `detailed::implied_vol`, 20k | 10.8 ms (1.85 M/s) | 1.03 ms (19.4 M/s) |
| `mc::european`, 1M paths | 11.9 ms (83.8 M paths/s) | 0.80 ms (1.25 G paths/s) |
| `mc::american_lsm`, 20k train + 20k eval, 50 steps | 36.6 ms | 29.3 ms |
| `dataset::generate`, 100k rows | 43.6 ms | 43.9 ms (not parallelised) |

Also in `benches/pricing.rs` (earlier run, serial): BSM price about 18 ns,
one implied-volatility solve about 118 ns, binomial American put 5.9 us /
112 us / 430 us at 100 / 500 / 1000 steps.

Notes: LSM parallelises only the evaluation phase, so it gains little; the
training regression is serial. Parallel speed-ups depend on the number of
cores and will be smaller on smaller machines.

## Python (`benchmarks/python_bench.py`, median of 15 runs, 1,000,000 rows)

Baseline: a NumPy + `scipy.special.ndtr` implementation of the same
Black-Scholes-Merton call formula (written for this benchmark; results agree
with quantkit to `5.7e-14` over all rows).

| call | all threads | `RAYON_NUM_THREADS=1` |
|---|---|---|
| `quantkit.bs_price_batch` | 23.9 ms (41.9 M rows/s) | 76.0 ms (13.2 M rows/s) |
| NumPy + SciPy baseline | 77.5 ms (12.9 M rows/s) | 91.9 ms (10.9 M rows/s) |
| `quantkit.bs_price_detailed` (status + messages) | 37.5 ms | 84.2 ms |
| `quantkit.bs_greeks_batch` | 43.8 ms | 125.3 ms |
| `quantkit.implied_vol_batch`, 200k rows | 16.8 ms | 98.1 ms |
| `quantkit.implied_vol_detailed`, 200k rows | 20.1 ms | 101.8 ms |
| `bs_price_batch`, 10 rows (per call) | 2.2 us | 2.6 us |
| NumPy + SciPy, 10 rows (per call) | 6.8 us | 6.3 us |

How to read this fairly:

* The NumPy baseline timing varied by about 20% between the two sessions
  (77 vs 92 ms), so the single-thread comparison (76 vs 77-92 ms) is
  "comparable", not a win. The multi-thread win (about 3x here) comes from
  using 32 cores; NumPy/SciPy above is single-threaded.
* The Rust core alone computes 1M prices in about 26 ms serially (the criterion
  figure scaled). The rest of the single-thread Python time is converting
  NumPy arrays into Rust vectors and back. A first version that built a
  struct per row took 94.7 ms single-threaded; the columnar path cut that to
  76 ms. That conversion cost is the main remaining overhead.
* Not measured: comparisons with `py_vollib_vectorized`, `pyvolr`, QuantLib,
  or any implied-volatility competitor. Their speed claims are theirs; none
  were reproduced here.

## Reproducing

```bash
cargo bench --bench throughput
cargo bench --bench throughput --features parallel
cd python && maturin develop --release && cd ..
python benchmarks/python_bench.py
RAYON_NUM_THREADS=1 python benchmarks/python_bench.py
```
