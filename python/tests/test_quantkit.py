import math

import numpy as np
import pytest

import quantkit as qk


def test_hull_example():
    # Hull, Options, Futures and Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=0.5.
    assert qk.bs_price(42, 40, 0.10, 0.20, 0.5) == pytest.approx(4.76, abs=0.005)
    assert qk.bs_price(42, 40, 0.10, 0.20, 0.5, is_call=False) == pytest.approx(0.81, abs=0.005)


def test_independent_python_reference():
    def ncdf(x):
        return 0.5 * math.erfc(-x / math.sqrt(2))

    s, k, r, q, v, t = 100.0, 95.0, 0.10, 0.05, 0.20, 0.5
    d1 = (math.log(s / k) + (r - q + v * v / 2) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    call = s * math.exp(-q * t) * ncdf(d1) - k * math.exp(-r * t) * ncdf(d2)
    assert qk.bs_price(s, k, r, v, t, True, q) == pytest.approx(call, abs=1e-11)


def test_greeks_keys_and_values():
    g = qk.bs_greeks(49, 50, 0.05, 0.20, 20 / 52)  # Hull's Greeks example
    assert set(g) == {"delta", "gamma", "vega", "theta", "rho"}
    assert g["delta"] == pytest.approx(0.522, abs=5e-4)
    assert g["theta"] == pytest.approx(-4.31, abs=5e-3)


def test_implied_vol_round_trip_and_hull():
    assert qk.implied_vol(1.875, 21, 20, 0.10, 0.25) == pytest.approx(0.235, abs=5e-4)
    price = qk.bs_price(100, 110, 0.03, 0.35, 0.75, False)
    assert qk.implied_vol(price, 100, 110, 0.03, 0.75, False) == pytest.approx(0.35, abs=1e-8)


def test_errors_are_value_errors_not_panics():
    with pytest.raises(qk.QuantError):
        qk.bs_price(100, 100, 0.05, -0.2, 1.0)
    with pytest.raises(ValueError):
        qk.bs_price(100, 100, 0.05, 0.2, 0.0)
    with pytest.raises(qk.QuantError):
        qk.bs_price(float("nan"), 100, 0.05, 0.2, 1.0)
    with pytest.raises(qk.QuantError):
        qk.implied_vol(-1.0, 100, 100, 0.05, 1.0)
    with pytest.raises(qk.QuantError):
        qk.historical_var(np.array([]), 0.95)


def test_binomial_and_mc_agree_with_bs():
    exact = qk.bs_price(100, 100, 0.05, 0.2, 1.0)
    assert qk.binomial_price(100, 100, 0.05, 0.2, 1.0, american=False, steps=2000) == pytest.approx(exact, abs=5e-3)
    price, se = qk.mc_european(100, 100, 0.05, 0.2, 1.0)
    assert abs(price - exact) < 4 * se
    # Same seed, same answer.
    assert qk.mc_european(100, 100, 0.05, 0.2, 1.0, seed=7) == qk.mc_european(100, 100, 0.05, 0.2, 1.0, seed=7)
    p, se = qk.mc_american(36, 40, 0.06, 0.2, 1.0, paths=50_000)
    tree = qk.binomial_price(36, 40, 0.06, 0.2, 1.0, is_call=False, american=True, steps=2000)
    assert abs(p - tree) < 0.08 and se > 0


def test_batch_matches_scalar_and_broadcasts():
    spots = np.linspace(80, 120, 9)
    out = qk.bs_price_batch(spots, 100.0, 0.03, 0.2, 1.0)
    assert out.shape == (9,) and out.dtype == np.float64
    for s, got in zip(spots, out):
        assert got == pytest.approx(qk.bs_price(s, 100, 0.03, 0.2, 1.0), abs=1e-12)
    g = qk.bs_greeks_batch(spots, 100.0, 0.03, 0.2, 1.0, is_call=False)
    assert g["delta"].shape == (9,)
    assert g["delta"][0] == pytest.approx(qk.bs_greeks(80, 100, 0.03, 0.2, 1.0, False)["delta"])
    ivs = qk.implied_vol_batch(out, spots, 100.0, 0.03, 1.0)
    np.testing.assert_allclose(ivs, 0.2, atol=1e-7)


def test_batch_error_modes():
    vols = np.array([0.2, -0.1, 0.3])
    with pytest.raises(qk.QuantError, match="row 1"):
        qk.bs_price_batch(100.0, 100.0, 0.03, vols, 1.0)
    out = qk.bs_price_batch(100.0, 100.0, 0.03, vols, 1.0, errors="nan")
    assert np.isnan(out[1]) and not np.isnan(out[0]) and not np.isnan(out[2])
    with pytest.raises(ValueError):
        qk.bs_price_batch(np.ones(3) * 100, np.ones(4) * 100, 0.03, 0.2, 1.0)
    with pytest.raises(ValueError):
        qk.bs_price_batch(100.0, 100.0, 0.03, 0.2, 1.0, errors="bogus")
    prices = np.array([10.0, -5.0])
    iv = qk.implied_vol_batch(prices, 100.0, 100.0, 0.03, 1.0, errors="nan")
    assert not np.isnan(iv[0]) and np.isnan(iv[1])


def test_risk_metrics_hand_computed():
    r = np.arange(-10, 10) / 100.0
    assert qk.historical_var(r, 0.95) == pytest.approx(0.10)
    assert qk.historical_cvar(r, 0.90) == pytest.approx(0.095)
    assert qk.max_drawdown(np.array([0.10, -0.20, 0.05, -0.10])) == pytest.approx(0.244)
    assert qk.sortino_ratio(np.array([0.02, -0.01, 0.03, -0.02]), 0.0, 1.0) == pytest.approx(0.4472135954999579)
    # Non-contiguous input works too.
    assert qk.historical_var(np.arange(-20, 20)[::2] / 100.0, 0.95) == pytest.approx(0.20)


def test_svi_fit_recovers_parameters():
    a, b, rho, m, sigma = 0.04, 0.4, -0.4, 0.05, 0.1
    ks = np.linspace(-1, 1, 41)
    ws = a + b * (rho * (ks - m) + np.sqrt((ks - m) ** 2 + sigma**2))
    fit = qk.svi_fit(ks, ws)
    assert fit["rmse"] < 1e-9
    for name, truth in zip(("a", "b", "rho", "m", "sigma"), (a, b, rho, m, sigma)):
        assert fit[name] == pytest.approx(truth, abs=1e-4)


def test_vol_surface():
    s = qk.VolSurface(
        100.0,
        [(0.5, [100.0], [0.20]), (1.5, [100.0], [0.30])],
    )
    assert s.vol(100.0, 1.0) == pytest.approx(math.sqrt(0.0775), abs=1e-12)
    assert s.check_arbitrage() == {"calendar": 0, "butterfly": 0}
    bad = qk.VolSurface(100.0, [(0.5, [100.0], [0.40]), (1.0, [100.0], [0.20])])
    assert bad.check_arbitrage()["calendar"] == 1
    with pytest.raises(qk.QuantError):
        qk.VolSurface(100.0, [(1.0, [100.0, 90.0], [0.2, 0.2])])
