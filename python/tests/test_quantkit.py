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
    g = qk.bs_greeks(49, 50, 0.05, 0.20, 20 / 52)
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
        qk.bs_price(1e300, 100, 0.05, 0.2, 1.0)  # outside the supported domain
    with pytest.raises(qk.QuantError):
        qk.implied_vol(-1.0, 100, 100, 0.05, 1.0)
    with pytest.raises(qk.QuantError):
        qk.historical_var(np.array([]), 0.95)
    with pytest.raises(qk.QuantError):
        qk.binomial_price(100, 100, 0.05, 0.2, 1.0, steps=10**12)


def test_binomial_and_mc_agree_with_bs():
    exact = qk.bs_price(100, 100, 0.05, 0.2, 1.0)
    assert qk.binomial_price(100, 100, 0.05, 0.2, 1.0, american=False, steps=2000) == pytest.approx(exact, abs=5e-3)
    price, se = qk.mc_european(100, 100, 0.05, 0.2, 1.0)
    assert abs(price - exact) < 4 * se
    assert qk.mc_european(100, 100, 0.05, 0.2, 1.0, seed=7) == qk.mc_european(100, 100, 0.05, 0.2, 1.0, seed=7)
    p, se = qk.mc_american(36, 40, 0.06, 0.2, 1.0, paths=50_000)
    tree = qk.binomial_price(36, 40, 0.06, 0.2, 1.0, is_call=False, american=True, steps=2000)
    assert abs(p - tree) < 0.08 and se > 0


def test_mc_golden_value_is_stable_across_python_and_rust():
    # Same configuration as the Rust regression-lock test (seed 42, 50k paths, antithetic).
    price, se = qk.mc_european(100, 100, 0.05, 0.2, 1.0, paths=50_000, seed=42, antithetic=True)
    assert np.float64(price).view(np.uint64) == 4622057817873776027
    assert np.float64(se).view(np.uint64) == 4586848783294312776


def test_mc_american_detailed_separates_training_and_evaluation():
    d = qk.mc_american_detailed(36, 40, 0.06, 0.2, 1.0, steps=25, train_paths=4000, eval_paths=4000)
    assert d["train_samples"] == 4000 and d["eval_samples"] == 2000  # antithetic pairs count once
    assert d["fitted_dates"] > 0 and d["std_error"] > 0
    assert d["in_sample_price"] != d["price"]
    d2 = qk.mc_american_detailed(36, 40, 0.06, 0.2, 1.0, steps=25, train_paths=4000, eval_paths=4000)
    assert d == d2


def test_scalar_in_scalar_out_and_nd_shapes():
    assert isinstance(qk.bs_price_batch(100.0, 100.0, 0.03, 0.2, 1.0), float)
    spots = np.linspace(80, 120, 12).reshape(3, 4)
    out = qk.bs_price_batch(spots, 100.0, 0.03, 0.2, 1.0)
    assert out.shape == (3, 4) and out.dtype == np.float64
    assert out[1, 2] == pytest.approx(qk.bs_price(spots[1, 2], 100, 0.03, 0.2, 1.0), abs=1e-12)
    g = qk.bs_greeks_batch(spots, 100.0, 0.03, 0.2, 1.0, is_call=False)
    assert g["delta"].shape == (3, 4)
    iv = qk.implied_vol_batch(out, spots, 100.0, 0.03, 1.0)
    np.testing.assert_allclose(iv, 0.2, atol=1e-7)
    assert iv.shape == (3, 4)


def test_noncontiguous_fortran_and_empty_inputs():
    base = np.linspace(80, 120, 20)
    strided = base[::2]
    out = qk.bs_price_batch(strided, 100.0, 0.03, 0.2, 1.0)
    np.testing.assert_allclose(out, [qk.bs_price(s, 100, 0.03, 0.2, 1.0) for s in strided], atol=1e-12)
    fortran = np.asfortranarray(base.reshape(4, 5))
    out_f = qk.bs_price_batch(fortran, 100.0, 0.03, 0.2, 1.0)
    np.testing.assert_allclose(out_f.ravel(), qk.bs_price_batch(fortran.ravel(), 100.0, 0.03, 0.2, 1.0))
    empty = qk.bs_price_batch(np.array([]), 100.0, 0.03, 0.2, 1.0)
    assert empty.shape == (0,)
    assert qk.implied_vol_batch(np.array([]), 100.0, 100.0, 0.03, 1.0).shape == (0,)


def test_batch_error_modes():
    vols = np.array([0.2, -0.1, 0.3])
    with pytest.raises(qk.QuantError, match="row 1"):
        qk.bs_price_batch(100.0, 100.0, 0.03, vols, 1.0)
    out = qk.bs_price_batch(100.0, 100.0, 0.03, vols, 1.0, errors="nan")
    assert np.isnan(out[1]) and not np.isnan(out[0]) and not np.isnan(out[2])
    with pytest.raises(ValueError, match="same shape"):
        qk.bs_price_batch(np.ones(3) * 100, np.ones(4) * 100, 0.03, 0.2, 1.0)
    with pytest.raises(ValueError, match="same shape"):
        qk.bs_price_batch(np.ones((2, 2)) * 100, np.ones(4) * 100, 0.03, 0.2, 1.0)
    with pytest.raises(ValueError):
        qk.bs_price_batch(100.0, 100.0, 0.03, 0.2, 1.0, errors="bogus")
    iv = qk.implied_vol_batch(np.array([10.0, -5.0]), 100.0, 100.0, 0.03, 1.0, errors="nan")
    assert not np.isnan(iv[0]) and np.isnan(iv[1])


def test_detailed_price_reports_why_a_row_failed():
    vols = np.array([0.2, -0.1, float("nan"), 0.3])
    d = qk.bs_price_detailed(100.0, 100.0, 0.03, vols, 1.0)
    st = d["status"]
    assert st.dtype == np.uint8
    assert list(st) == [qk.STATUS["Ok"], qk.STATUS["InvalidInput"], qk.STATUS["NonFinite"], qk.STATUS["Ok"]]
    assert np.isnan(d["price"][1]) and np.isnan(d["price"][2])
    assert d["messages"][0] is None and "volatility" in d["messages"][1]
    assert qk.status_name(int(st[2])) == "NonFinite"


def test_detailed_implied_vol_has_diagnostics_and_distinct_failure_kinds():
    base = qk.bs_price(100, 105, 0.03, 0.25, 0.75)
    prices = np.array([base, -1.0, float("nan"), 1e-300])
    strikes = np.array([105.0, 105.0, 105.0, 400.0])
    d = qk.implied_vol_detailed(prices, 100.0, strikes, 0.03, 0.75)
    st = d["status"]
    assert st[0] == qk.STATUS["Ok"] and d["vol"][0] == pytest.approx(0.25, abs=1e-8)
    assert d["residual"][0] <= 1e-9 and d["vega"][0] > 0 and d["iterations"][0] >= d["newton_steps"][0]
    assert st[1] == qk.STATUS["PriceOutOfBounds"]
    assert st[2] == qk.STATUS["NonFinite"]
    assert st[3] != qk.STATUS["Ok"] and np.isnan(d["vol"][3])
    # Tightening the iteration budget turns a solvable row into a visible failure.
    tight = qk.implied_vol_detailed(np.array([base]), 100.0, 105.0, 0.03, 0.75, max_iterations=1, tolerance=1e-15)
    assert tight["status"][0] == qk.STATUS["NoConvergence"]
    assert tight["iterations"][0] >= 1 and np.isfinite(tight["residual"][0])


def test_detailed_mc_rows_are_independent_and_reproducible():
    spots = np.array([100.0, 100.0, 100.0])
    a = qk.mc_european_detailed(spots, 100.0, 0.03, 0.2, 1.0, paths=20_000, seed=3)
    assert len(set(a["seed"].tolist())) == 3
    assert np.all(a["std_error"] > 0) and np.all(a["status"] == 0)
    b = qk.mc_european_detailed(spots, 100.0, 0.03, 0.2, 1.0, paths=20_000, seed=3)
    np.testing.assert_array_equal(a["price"], b["price"])
    exact = qk.bs_price(100, 100, 0.03, 0.2, 1.0)
    assert np.all(np.abs(a["price"] - exact) < 5 * a["std_error"])


def test_risk_metrics_hand_computed():
    r = np.arange(-10, 10) / 100.0
    assert qk.historical_var(r, 0.95) == pytest.approx(0.10)
    assert qk.historical_cvar(r, 0.90) == pytest.approx(0.095)
    assert qk.max_drawdown(np.array([0.10, -0.20, 0.05, -0.10])) == pytest.approx(0.244)
    assert qk.sortino_ratio(np.array([0.02, -0.01, 0.03, -0.02]), 0.0, 1.0) == pytest.approx(0.4472135954999579)
    assert qk.historical_var(np.arange(-20, 20)[::2] / 100.0, 0.95) == pytest.approx(0.20)
    with pytest.raises(qk.QuantError):
        qk.max_drawdown(np.array([1e308, 1e308, -0.5]))


def test_svi_fit_recovers_parameters():
    a, b, rho, m, sigma = 0.04, 0.4, -0.4, 0.05, 0.1
    ks = np.linspace(-1, 1, 41)
    ws = a + b * (rho * (ks - m) + np.sqrt((ks - m) ** 2 + sigma**2))
    fit = qk.svi_fit(ks, ws)
    assert fit["rmse"] < 1e-9 and fit["max_abs_residual"] < 1e-8
    for name, truth in zip(("a", "b", "rho", "m", "sigma"), (a, b, rho, m, sigma)):
        assert fit[name] == pytest.approx(truth, abs=1e-4)


def test_vol_surface_and_scoped_arbitrage_reports():
    s = qk.VolSurface(100.0, [(0.5, [100.0], [0.20]), (1.5, [100.0], [0.30])])
    assert s.vol(100.0, 1.0) == pytest.approx(math.sqrt(0.0775), abs=1e-12)
    r = s.check_arbitrage()
    assert (r["calendar"], r["butterfly"], r["scope"]) == (0, 0, "quoted_nodes")
    assert s.check_arbitrage_sampled()["scope"].startswith("sampled_grid")
    bad = qk.VolSurface(100.0, [(0.5, [100.0], [0.40]), (1.0, [100.0], [0.20])])
    assert bad.check_arbitrage()["calendar"] == 1
    with pytest.raises(qk.QuantError):
        qk.VolSurface(100.0, [(1.0, [100.0, 90.0], [0.2, 0.2])])


def test_make_dataset_is_reproducible_and_audited():
    a = qk.make_dataset(rows=300, seed=11)
    b = qk.make_dataset(rows=300, seed=11)
    c = qk.make_dataset(rows=300, seed=12)
    assert a["metadata"]["checksum"] == b["metadata"]["checksum"] != c["metadata"]["checksum"]
    cols = a["columns"]
    n = 300 + a["metadata"]["boundary_rows"]
    assert all(len(v) == n for v in cols.values())
    assert a["metadata"]["invariants"]["all_passed"] is True
    # Labels agree with the scalar API.
    i = 150 + a["metadata"]["boundary_rows"]
    price = qk.bs_price(cols["spot"][i], cols["strike"][i], cols["rate"][i], cols["volatility"][i],
                        cols["expiry"][i], bool(cols["is_call"][i]), cols["dividend_yield"][i])
    assert price == pytest.approx(cols["price"][i], abs=1e-12)
    # Same checksum as the Rust regression-lock test.
    locked = qk.make_dataset(rows=200, seed=123)
    assert locked["metadata"]["checksum"] == 5032231679998956232
    with pytest.raises(qk.QuantError):
        qk.make_dataset(rows=10, volatility=(1e-12, 1.0))


def test_thread_pool_setting():
    with pytest.raises(ValueError):
        qk.set_num_threads(0)
    try:
        qk.set_num_threads(4)
    except RuntimeError:
        pass  # pool was already initialised by an earlier parallel call
