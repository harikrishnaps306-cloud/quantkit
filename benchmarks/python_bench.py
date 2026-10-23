"""Python-level throughput measurements for quantkit.

Run (after `maturin develop --release` in python/):

    python benchmarks/python_bench.py
    RAYON_NUM_THREADS=1 python benchmarks/python_bench.py

It reports medians over repeated runs, the NumPy+SciPy baseline used for the
Black-Scholes price comparison, and the library/interpreter versions. Numbers
depend on the machine; see docs/BENCHMARKS.md for how to read them.
"""

import os
import platform
import statistics
import sys
import time

import numpy as np

import quantkit as qk

try:
    from scipy.special import ndtr
except ImportError:  # the comparison is skipped without SciPy
    ndtr = None


def bench(fn, repeats=15, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times)


def numpy_bsm_call(s, k, r, q, v, t):
    """Reference NumPy + SciPy vectorised Black-Scholes-Merton call price."""
    sq = v * np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * v * v) * t) / sq
    d2 = d1 - sq
    return s * np.exp(-q * t) * ndtr(d1) - k * np.exp(-r * t) * ndtr(d2)


def main():
    n = 1_000_000
    rng = np.random.default_rng(0)
    s = rng.uniform(50, 150, n)
    k = rng.uniform(60, 140, n)
    r = rng.uniform(0.0, 0.06, n)
    q = rng.uniform(0.0, 0.03, n)
    v = rng.uniform(0.1, 0.8, n)
    t = rng.uniform(0.05, 3.0, n)

    print("python", sys.version.split()[0], "| numpy", np.__version__, "| quantkit", qk.__version__)
    if ndtr is not None:
        import scipy

        print("scipy", scipy.__version__)
    print("platform", platform.platform(), "| cpu count", os.cpu_count())
    print("RAYON_NUM_THREADS =", os.environ.get("RAYON_NUM_THREADS", "(unset)"))
    print(f"rows per call: {n:,}\n")

    med, best = bench(lambda: qk.bs_price_batch(s, k, r, v, t, True, q))
    print(f"quantkit bs_price_batch      median {med*1e3:8.2f} ms  best {best*1e3:8.2f} ms  ({n/med/1e6:7.1f} Mrows/s)")

    if ndtr is not None:
        a = qk.bs_price_batch(s, k, r, v, t, True, q)
        b = numpy_bsm_call(s, k, r, q, v, t)
        max_diff = float(np.max(np.abs(a - b)))
        med_np, best_np = bench(lambda: numpy_bsm_call(s, k, r, q, v, t))
        print(f"numpy+scipy.special.ndtr     median {med_np*1e3:8.2f} ms  best {best_np*1e3:8.2f} ms  ({n/med_np/1e6:7.1f} Mrows/s)")
        print(f"  max |quantkit - numpy/scipy| over {n:,} rows: {max_diff:.2e}")

    med, _ = bench(lambda: qk.bs_price_detailed(s, k, r, v, t, True, q), repeats=7)
    print(f"quantkit bs_price_detailed   median {med*1e3:8.2f} ms  ({n/med/1e6:7.1f} Mrows/s)  [adds status codes + messages]")

    med, _ = bench(lambda: qk.bs_greeks_batch(s, k, r, v, t, True, q), repeats=7)
    print(f"quantkit bs_greeks_batch     median {med*1e3:8.2f} ms  ({n/med/1e6:7.1f} Mrows/s)")

    # Implied volatility on 200k rows built from known vols.
    m = 200_000
    prices = qk.bs_price_batch(s[:m], k[:m], r[:m], v[:m], t[:m], True, q[:m])
    med, _ = bench(lambda: qk.implied_vol_batch(prices, s[:m], k[:m], r[:m], t[:m], True, q[:m], "nan"), repeats=7)
    print(f"quantkit implied_vol_batch   median {med*1e3:8.2f} ms  ({m/med/1e6:7.2f} Mrows/s, {m:,} rows)")
    med, _ = bench(lambda: qk.implied_vol_detailed(prices, s[:m], k[:m], r[:m], t[:m], True, q[:m]), repeats=5)
    print(f"quantkit implied_vol_detailed median {med*1e3:7.2f} ms  ({m/med/1e6:7.2f} Mrows/s)  [adds residual, iterations, vega]")

    # Tiny batches: per-call overhead.
    s1, k1, r1, v1, t1, q1 = s[:10], k[:10], r[:10], v[:10], t[:10], q[:10]
    med, _ = bench(lambda: qk.bs_price_batch(s1, k1, r1, v1, t1, True, q1), repeats=2000, warmup=100)
    print(f"\nquantkit bs_price_batch, 10 rows: median {med*1e6:6.1f} us per call (overhead-dominated)")
    if ndtr is not None:
        med, _ = bench(lambda: numpy_bsm_call(s1, k1, r1, q1, v1, t1), repeats=2000, warmup=100)
        print(f"numpy+scipy, 10 rows:             median {med*1e6:6.1f} us per call")


if __name__ == "__main__":
    main()
