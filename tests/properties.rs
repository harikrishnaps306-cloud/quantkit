//! Property-based tests (proptest) for the pricing and risk modules.

use proptest::prelude::*;
use quantkit::{binomial, bsm, greeks, implied_vol, risk, ExerciseStyle, OptionParams, OptionType};

#[allow(clippy::too_many_arguments)]
fn params(s: f64, k: f64, r: f64, q: f64, v: f64, t: f64) -> OptionParams {
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn put_call_parity(
        s in 10.0..300.0f64, k in 10.0..300.0f64, r in -0.02..0.15f64,
        q in 0.0..0.08f64, v in 0.05..1.5f64, t in 0.02..3.0f64,
    ) {
        let p = params(s, k, r, q, v, t);
        let lhs = bsm::call_price(&p) - bsm::put_price(&p);
        let rhs = s * (-q * t).exp() - k * (-r * t).exp();
        prop_assert!((lhs - rhs).abs() < 1e-9 * (s + k), "{lhs} vs {rhs}");
    }

    #[test]
    fn call_price_monotonic_in_spot(
        s in 10.0..300.0f64, bump in 0.0..50.0f64, k in 10.0..300.0f64,
        r in 0.0..0.1f64, q in 0.0..0.05f64, v in 0.05..1.0f64, t in 0.02..3.0f64,
    ) {
        let lo = bsm::call_price(&params(s, k, r, q, v, t));
        let hi = bsm::call_price(&params(s + bump, k, r, q, v, t));
        prop_assert!(hi >= lo - 1e-12 * (s + k));
    }

    #[test]
    fn call_delta_between_zero_and_one(
        s in 10.0..300.0f64, k in 10.0..300.0f64, r in -0.02..0.1f64,
        q in 0.0..0.08f64, v in 0.05..1.5f64, t in 0.02..3.0f64,
    ) {
        let g = greeks::analytic(&params(s, k, r, q, v, t), OptionType::Call);
        prop_assert!((0.0..=1.0).contains(&g.delta), "delta {}", g.delta);
        let gp = greeks::analytic(&params(s, k, r, q, v, t), OptionType::Put);
        prop_assert!((-1.0..=0.0).contains(&gp.delta), "put delta {}", gp.delta);
    }

    #[test]
    fn gamma_and_vega_non_negative(
        s in 10.0..300.0f64, k in 10.0..300.0f64, r in 0.0..0.1f64,
        v in 0.05..1.5f64, t in 0.02..3.0f64,
    ) {
        let g = greeks::analytic(&params(s, k, r, 0.0, v, t), OptionType::Call);
        prop_assert!(g.gamma >= 0.0 && g.vega >= 0.0);
    }

    #[test]
    fn american_at_least_european(
        s in 20.0..200.0f64, k in 20.0..200.0f64, r in 0.0..0.1f64,
        q in 0.0..0.05f64, v in 0.1..0.8f64, t in 0.05..2.0f64, is_call in any::<bool>(),
    ) {
        let kind = if is_call { OptionType::Call } else { OptionType::Put };
        let p = params(s, k, r, q, v, t);
        let am = binomial::price(&p, kind, ExerciseStyle::American, 60).unwrap();
        let eu = binomial::price(&p, kind, ExerciseStyle::European, 60).unwrap();
        prop_assert!(am >= eu - 1e-12, "{am} < {eu}");
    }

    #[test]
    fn implied_vol_round_trips(
        s in 50.0..200.0f64, moneyness in 0.7..1.4f64, r in 0.0..0.08f64,
        q in 0.0..0.04f64, v in 0.1..1.2f64, t in 0.1..2.0f64, is_call in any::<bool>(),
    ) {
        let kind = if is_call { OptionType::Call } else { OptionType::Put };
        let p = params(s, s * moneyness, r, q, v, t);
        let price = bsm::price(&p, kind);
        let iv = implied_vol::solve(p.contract(), kind, price);
        prop_assume!(iv.is_ok()); // price indistinguishable from a bound
        let iv = iv.unwrap();
        // The solved vol must always reprice the option...
        let back = bsm::price(&p.with_vol(iv).unwrap(), kind);
        prop_assert!((back - price).abs() <= 1e-9 * price, "{back} vs {price}");
        // ...and must recover the input vol whenever the problem is
        // well-conditioned (a deep-ITM price equal to intrinsic value to
        // machine precision says almost nothing about volatility).
        let vega = greeks::analytic(&p, kind).vega;
        if vega > 1e-3 {
            prop_assert!((iv - v).abs() < 1e-5, "iv {iv} vs {v} (vega {vega})");
        }
    }

    #[test]
    fn bsm_price_within_no_arbitrage_bounds(
        s in 10.0..300.0f64, k in 10.0..300.0f64, r in 0.0..0.1f64,
        q in 0.0..0.05f64, v in 0.05..1.5f64, t in 0.02..3.0f64,
    ) {
        let p = params(s, k, r, q, v, t);
        let (ds, dk) = (s * (-q * t).exp(), k * (-r * t).exp());
        let c = bsm::call_price(&p);
        prop_assert!(c >= (ds - dk).max(0.0) - 1e-9 * s && c <= ds + 1e-9 * s);
    }

    #[test]
    fn cvar_at_least_var_and_drawdown_in_unit_interval(
        rets in prop::collection::vec(-0.5..0.5f64, 1..200), c in 0.5..0.999f64,
    ) {
        let var = risk::historical_var(&rets, c).unwrap();
        let cvar = risk::historical_cvar(&rets, c).unwrap();
        prop_assert!(cvar >= var - 1e-12);
        let dd = risk::max_drawdown(&rets).unwrap();
        prop_assert!((0.0..=1.0).contains(&dd));
    }

    #[test]
    fn historical_var_monotone_in_confidence(
        rets in prop::collection::vec(-0.5..0.5f64, 1..200),
        c1 in 0.5..0.99f64, dc in 0.0..0.009f64,
    ) {
        let lo = risk::historical_var(&rets, c1).unwrap();
        let hi = risk::historical_var(&rets, c1 + dc).unwrap();
        prop_assert!(hi >= lo - 1e-12);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(40))]

    /// Noise-free SVI data from admissible parameters must be fitted to
    /// high accuracy (the true parameters are a feasible optimum).
    #[test]
    fn svi_fit_reproduces_noise_free_smiles(
        a in 0.01..0.08f64, b in 0.1..0.8f64, rho in -0.8..0.8f64,
        m in -0.2..0.2f64, sigma in 0.05..0.4f64,
    ) {
        use quantkit::surface::{fit_svi, SviParams};
        let truth = SviParams::new(a, b, rho, m, sigma).unwrap();
        let ks: Vec<f64> = (-20..=20).map(|i| f64::from(i) / 20.0).collect();
        let ws: Vec<f64> = ks.iter().map(|&k| truth.total_variance(k)).collect();
        let fit = fit_svi(&ks, &ws).unwrap();
        prop_assert!(fit.rmse < 1e-7, "rmse {}", fit.rmse);
    }

    #[test]
    fn mc_european_within_five_standard_errors_of_bsm(
        s in 50.0..150.0f64, m in 0.8..1.25f64, r in 0.0..0.08f64,
        v in 0.1..0.6f64, t in 0.1..2.0f64, seed in 0u64..1000,
    ) {
        use quantkit::mc::{european, McConfig};
        // Strikes near the money only: for deep out-of-the-money options
        // nearly every path pays zero and the sample standard error is
        // meaningless (a known limit of plain Monte Carlo).
        let p = params(s, s * m, r, 0.0, v, t);
        let est = european(&p, OptionType::Call, McConfig { paths: 40_000, seed, antithetic: true }).unwrap();
        let exact = bsm::call_price(&p);
        prop_assert!((est.price - exact).abs() <= 5.0 * est.std_error + 1e-9,
            "{} vs {exact} (se {})", est.price, est.std_error);
    }
}
