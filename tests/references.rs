//! Cross-checks against relations that are derived *independently* of the
//! implementation under test: other closed forms, exact symmetries, limits,
//! and finite differences of prices. No expected value here is produced by
//! the code being tested.

use proptest::prelude::*;
use quantkit::implied_vol::{solve_detailed, SolverConfig};
use quantkit::surface::SviParams;
use quantkit::{binomial, bsm, greeks, ExerciseStyle, OptionParams, OptionType, QuantError};

fn opt(s: f64, k: f64, r: f64, q: f64, v: f64, t: f64) -> OptionParams {
    OptionParams::builder()
        .spot(s)
        .strike(k)
        .rate(r)
        .dividend_yield(q)
        .volatility(v)
        .expiry(t)
        .build()
        .unwrap()
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

/// With r = q = 0 and S = K the BSM call collapses to `S * erf(sigma sqrt(T) / (2 sqrt 2))`
/// (using `Phi(x) = (1 + erf(x / sqrt 2)) / 2`). That is a different closed
/// form from the d1/d2 formula the library evaluates, via `erf` not `erfc`.
#[test]
fn atm_zero_rate_collapses_to_an_erf_closed_form() {
    for &(s, v, t) in &[
        (100.0, 0.2, 1.0),
        (37.5, 0.8, 0.25),
        (1e4, 0.05, 5.0),
        (3.0, 2.0, 0.01),
    ] {
        let x = v * f64::sqrt(t) / (2.0 * std::f64::consts::SQRT_2);
        let reference = s * libm::erf(x);
        let p = opt(s, s, 0.0, 0.0, v, t);
        assert!(
            rel(bsm::call_price(&p), reference) < 1e-13,
            "s={s} v={v} t={t}"
        );
        assert!(rel(bsm::put_price(&p), reference) < 1e-13);
    }
}

/// Exact symmetry (McDonald-Schroder / Bjerksund-Stensland):
/// `C(S, K, r, q) = P(K, S, q, r)` with the same volatility and expiry.
/// A bug in the dividend yield or discounting breaks this immediately.
#[test]
fn call_put_symmetry_swaps_rate_and_dividend_yield() {
    let cases = [
        (100.0, 95.0, 0.03, 0.01, 0.25, 0.75),
        (50.0, 80.0, 0.08, 0.02, 0.4, 2.0),
        (120.0, 100.0, -0.01, 0.04, 0.15, 0.1),
        (10.0, 10.0, 0.0, 0.0, 0.3, 1.0),
    ];
    for (s, k, r, q, v, t) in cases {
        let call = bsm::call_price(&opt(s, k, r, q, v, t));
        let mirrored_put = bsm::put_price(&opt(k, s, q, r, v, t));
        assert!(
            rel(call, mirrored_put) < 1e-12,
            "{s} {k} {r} {q}: {call} vs {mirrored_put}"
        );
    }
}

/// Prices are homogeneous of degree one in (S, K).
#[test]
fn price_is_homogeneous_in_spot_and_strike() {
    let base = opt(40.0, 45.0, 0.05, 0.02, 0.3, 1.5);
    for lambda in [1e-3, 0.37, 12.0, 1e6] {
        let scaled = opt(40.0 * lambda, 45.0 * lambda, 0.05, 0.02, 0.3, 1.5);
        for kind in [OptionType::Call, OptionType::Put] {
            assert!(rel(bsm::price(&scaled, kind), lambda * bsm::price(&base, kind)) < 1e-12);
        }
    }
}

/// Deterministic limit: at the smallest supported volatility the price is the
/// discounted forward intrinsic value (the documented behaviour at "zero"
/// volatility, which the library does not accept as an input).
#[test]
fn smallest_supported_vol_gives_forward_intrinsic_value() {
    let (s, k, r, q, t) = (110.0, 100.0, 0.04, 0.01, 1.0);
    let p = opt(s, k, r, q, quantkit::limits::MIN_VOL, t);
    let call = s * (-q * t).exp() - k * (-r * t).exp();
    assert!((bsm::call_price(&p) - call).abs() < 1e-9);
    assert!(bsm::put_price(&p) < 1e-9);
}

/// Analytic Greeks against 5-point central differences of *prices* only.
/// The stencil has O(h^4) error, so the tolerances below are tight.
#[test]
fn analytic_greeks_match_high_order_finite_differences() {
    let stencil = |f: &dyn Fn(f64) -> f64, x: f64, h: f64| {
        (-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h)
    };
    let stencil2 = |f: &dyn Fn(f64) -> f64, x: f64, h: f64| {
        (-f(x + 2.0 * h) + 16.0 * f(x + h) - 30.0 * f(x) + 16.0 * f(x - h) - f(x - 2.0 * h))
            / (12.0 * h * h)
    };
    for &(s, k, r, q, v, t) in &[
        (100.0, 100.0, 0.05, 0.0, 0.2, 1.0),
        (100.0, 90.0, 0.03, 0.02, 0.35, 0.5),
        (80.0, 100.0, 0.01, 0.0, 0.6, 2.0),
    ] {
        for kind in [OptionType::Call, OptionType::Put] {
            let g = greeks::analytic(&opt(s, k, r, q, v, t), kind);
            let price = |s: f64, r: f64, v: f64, t: f64| bsm::price(&opt(s, k, r, q, v, t), kind);
            let delta = stencil(&|x| price(x, r, v, t), s, 1e-3 * s);
            let gamma = stencil2(&|x| price(x, r, v, t), s, 1e-3 * s);
            let vega = stencil(&|x| price(s, r, x, t), v, 1e-3);
            let rho = stencil(&|x| price(s, x, v, t), r, 1e-4);
            let theta = -stencil(&|x| price(s, r, v, x), t, 1e-3);
            let ctx = format!("{kind:?} {s} {k} {r} {q} {v} {t}");
            assert!(
                (g.delta - delta).abs() < 1e-7,
                "delta {ctx}: {} vs {delta}",
                g.delta
            );
            assert!(
                (g.gamma - gamma).abs() < 1e-7,
                "gamma {ctx}: {} vs {gamma}",
                g.gamma
            );
            assert!(
                (g.vega - vega).abs() < 1e-6 * (1.0 + vega.abs()),
                "vega {ctx}"
            );
            assert!((g.rho - rho).abs() < 1e-6 * (1.0 + rho.abs()), "rho {ctx}");
            assert!(
                (g.theta - theta).abs() < 1e-6 * (1.0 + theta.abs()),
                "theta {ctx}"
            );
        }
    }
}

/// With r = 0 an American put is worth the same as the European put (there
/// is no benefit to exercising early), and with q = 0 an American call equals
/// the European call. Both are exact properties of the discrete tree too.
#[test]
fn american_equals_european_where_early_exercise_has_no_value() {
    for steps in [1usize, 7, 64, 300] {
        let put = opt(100.0, 110.0, 0.0, 0.0, 0.3, 1.0);
        let am = binomial::price(&put, OptionType::Put, ExerciseStyle::American, steps).unwrap();
        let eu = binomial::price(&put, OptionType::Put, ExerciseStyle::European, steps).unwrap();
        assert!((am - eu).abs() < 1e-12, "r=0 put, steps {steps}");
        let call = opt(100.0, 90.0, 0.05, 0.0, 0.3, 1.0);
        let am = binomial::price(&call, OptionType::Call, ExerciseStyle::American, steps).unwrap();
        let eu = binomial::price(&call, OptionType::Call, ExerciseStyle::European, steps).unwrap();
        assert!((am - eu).abs() < 1e-12, "q=0 call, steps {steps}");
    }
}

/// McKean's perpetual American put (q = 0): with gamma = 2r / sigma^2 and
/// exercise boundary S* = K gamma / (1 + gamma), the value for S > S* is
/// `(K - S*) (S / S*)^(-gamma)`. A long-dated finite-horizon put can only be
/// worth *less* (fewer exercise opportunities) and converges to it as T grows.
#[test]
fn long_dated_american_put_approaches_the_perpetual_closed_form() {
    let (s, k, r, v): (f64, f64, f64, f64) = (100.0, 100.0, 0.1, 0.4);
    let gamma = 2.0 * r / (v * v);
    let boundary = k * gamma / (1.0 + gamma);
    let perpetual = (k - boundary) * (s / boundary).powf(-gamma);
    let value = |t: f64| {
        binomial::price(
            &opt(s, k, r, 0.0, v, t),
            OptionType::Put,
            ExerciseStyle::American,
            4000,
        )
        .unwrap()
    };
    let (v20, v60) = (value(20.0), value(60.0));
    assert!(v20 <= v60 + 1e-9, "value must not decrease with maturity");
    assert!(
        v60 <= perpetual * (1.0 + 2e-3),
        "finite put {v60} above perpetual {perpetual}"
    );
    assert!(
        rel(v60, perpetual) < 5e-3,
        "finite put {v60} vs perpetual {perpetual}"
    );
}

/// The SVI butterfly density `g(k)` from Gatheral-Jacquier must reproduce the
/// risk-neutral density obtained by differentiating Black call prices twice:
/// `p(k) = e^{-k} (c'' - c')` with `c(k)` the call price per unit forward,
/// and also `p(k) = g(k) / sqrt(2 pi w) * exp(-d2^2 / 2)`.
#[test]
fn svi_butterfly_g_matches_the_density_from_call_price_derivatives() {
    let svi = SviParams::new(0.04, 0.4, -0.4, 0.05, 0.1).unwrap();
    let phi = |x: f64| 0.5 * libm::erfc(-x / std::f64::consts::SQRT_2);
    let call = |k: f64| {
        let w = svi.total_variance(k);
        let sw = w.sqrt();
        let d1 = -k / sw + 0.5 * sw;
        let d2 = d1 - sw;
        phi(d1) - k.exp() * phi(d2)
    };
    // 5-point stencils (O(h^4) truncation) keep finite-difference error far
    // below the tolerance, so a mismatch would be a real formula error.
    let h = 5e-3;
    for &k in &[-0.6, -0.2, 0.0, 0.15, 0.5] {
        let c1 = (-call(k + 2.0 * h) + 8.0 * call(k + h) - 8.0 * call(k - h) + call(k - 2.0 * h))
            / (12.0 * h);
        let c2 = (-call(k + 2.0 * h) + 16.0 * call(k + h) - 30.0 * call(k) + 16.0 * call(k - h)
            - call(k - 2.0 * h))
            / (12.0 * h * h);
        let density_fd = (-k).exp() * (c2 - c1);
        let w = svi.total_variance(k);
        let d2 = -k / w.sqrt() - 0.5 * w.sqrt();
        let density_g =
            svi.butterfly_g(k) / (2.0 * std::f64::consts::PI * w).sqrt() * (-0.5 * d2 * d2).exp();
        assert!(
            (density_fd - density_g).abs() < 2e-6 * (1.0 + density_g.abs()),
            "k={k}: finite-difference density {density_fd} vs g-formula {density_g}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// For *any* market price strictly inside the no-arbitrage range, the
    /// solver either returns a volatility whose model price is within the
    /// documented tolerance (the guarantee), or a typed error. It never
    /// returns an arbitrary value. Volatility recovery is only demanded to
    /// the accuracy the residual/vega conditioning allows.
    #[test]
    fn iv_solution_honours_its_documented_guarantee(
        s in 20.0..500.0f64, m in 0.25..4.0f64, r in -0.02..0.1f64, q in 0.0..0.06f64,
        t in 0.002..10.0f64, frac in 0.0..1.0f64, is_call in any::<bool>(),
    ) {
        let kind = if is_call { OptionType::Call } else { OptionType::Put };
        let contract = OptionParams::builder()
            .spot(s).strike(s * m).rate(r).dividend_yield(q).expiry(t)
            .build_contract().unwrap();
        let ds = s * (-q * t).exp();
        let dk = s * m * (-r * t).exp();
        let (lower, upper) = if is_call { ((ds - dk).max(0.0), ds) } else { ((dk - ds).max(0.0), dk) };
        let price = lower + frac * (upper - lower);
        let cfg = SolverConfig::default();
        match solve_detailed(&contract, kind, price, cfg) {
            Ok(sol) => {
                let back = bsm::price(&contract.with_volatility(sol.vol).unwrap(), kind);
                let tol = cfg.tolerance.max(8.0 * f64::EPSILON) * price;
                prop_assert!((back - price).abs() <= tol * (1.0 + 1e-9),
                    "residual {} > tol {tol}", (back - price).abs());
                prop_assert!((sol.residual - (back - price).abs()).abs() <= 1e-9 * price + 1e-300);
                prop_assert!(sol.vol >= quantkit::limits::MIN_VOL && sol.vol <= quantkit::limits::MAX_VOL);
                prop_assert!(sol.vega.is_finite() && sol.vega >= 0.0);
            }
            Err(QuantError::PriceOutOfBounds { .. }) | Err(QuantError::NoConvergence { .. }) => {}
            Err(other) => prop_assert!(false, "unexpected error kind: {other:?}"),
        }
    }

    /// Volatility round trip with a conditioning-aware tolerance:
    /// |iv - sigma| * vega is bounded by the price tolerance.
    #[test]
    fn iv_round_trip_error_is_bounded_by_residual_over_vega(
        s in 20.0..500.0f64, m in 0.5..2.0f64, r in 0.0..0.08f64,
        t in 0.05..5.0f64, v in 0.05..2.0f64, is_call in any::<bool>(),
    ) {
        let kind = if is_call { OptionType::Call } else { OptionType::Put };
        let p = opt(s, s * m, r, 0.0, v, t);
        let price = bsm::price(&p, kind);
        let sol = solve_detailed(p.contract(), kind, price, SolverConfig::default());
        prop_assume!(sol.is_ok());
        let sol = sol.unwrap();
        let vega_true = greeks::analytic(&p, kind).vega;
        let vega_min = vega_true.min(sol.vega);
        if vega_min > 0.0 {
            let tol = 8.0 * f64::EPSILON.max(1e-12) * price;
            prop_assert!((sol.vol - v).abs() * vega_min <= 4.0 * tol + 1e-12 * price,
                "vol error {} * vega {vega_min} > price tolerance {tol}", (sol.vol - v).abs());
        }
    }
}
