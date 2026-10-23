"""Measures the relative accuracy of quantkit's closed-form BSM call price for
options far out of the money, against 60-digit arithmetic (mpmath).

Requires: pip install mpmath, and the quantkit Python wheel.

    python benchmarks/bsm_tail_accuracy.py

For an out-of-the-money call the price is a difference of two nearly equal
terms, S e^{-qT} N(d1) - K e^{-rT} N(d2), so relative accuracy degrades as
|d1| grows even though each term is accurate. The absolute error stays
negligible (many orders of magnitude below the spot).
"""

import math

from mpmath import erfc, log, mp, mpf, sqrt

import quantkit as qk

mp.dps = 60


def reference_call(s, k, v, t):
    s, k, v, t = map(mpf, (s, k, v, t))
    sq = v * sqrt(t)
    d1 = (log(s / k) + sq * sq / 2) / sq
    d2 = d1 - sq

    def cdf(z):
        return erfc(-z / sqrt(2)) / 2

    return s * cdf(d1) - k * cdf(d2)


def main():
    v, t = 0.2, 0.01
    sq = v * math.sqrt(t)
    print("S = 100, r = q = 0, sigma = 0.2, T = 0.01; strike chosen so that d1 = -x")
    print(f"{'|d1|':>6} {'price':>14} {'relative error':>16}")
    for x in (2.0, 5.0, 10.0, 20.0, 30.0, 38.0):
        k = 100.0 * math.exp(x * sq + 0.5 * sq * sq)
        got = qk.bs_price(100.0, k, 0.0, v, t)
        ref = reference_call(100.0, k, v, t)
        rel = abs(mpf(got) - ref) / ref
        print(f"{x:6.1f} {got:14.6e} {float(rel):16.2e}")


if __name__ == "__main__":
    main()
