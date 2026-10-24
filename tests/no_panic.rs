//! Robustness tests: whatever garbage goes in, the public API must return
//! `Ok` or `Err`, never panic. Successful Result-valued risk and quantile
//! calculations must also be finite. Raw unchecked helpers may return NaN.

use proptest::prelude::*;
use quantkit::mc::{self, McConfig};
use quantkit::surface::{fit_svi, SurfaceSlice, SviParams, VolSurface};
use quantkit::{
    batch, binomial, bsm, dist, greeks, implied_vol, risk, ExerciseStyle, OptionParams, OptionType,
};

/// Mostly ordinary floats, with a heavy dose of nasty ones.
fn wild() -> impl Strategy<Value = f64> {
    prop_oneof![
        3 => -1.0e3..1.0e3f64,
        2 => any::<f64>(),
        1 => Just(f64::NAN),
        1 => Just(f64::INFINITY),
        1 => Just(f64::NEG_INFINITY),
        1 => Just(0.0),
        1 => Just(-0.0),
        1 => Just(f64::MIN_POSITIVE),
        1 => Just(f64::MAX),
        1 => Just(f64::EPSILON),
    ]
}

fn wild_vec(max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(wild(), 0..max_len)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(std::env::var("PROPTEST_CASES").ok().and_then(|s| s.parse().ok()).unwrap_or(400)))]

    #[test]
    fn pricing_pipeline_never_panics(
        s in wild(), k in wild(), r in wild(), q in wild(), v in wild(), t in wild(),
        price in wild(), steps in prop_oneof![Just(0usize), 1usize..40, Just(usize::MAX)],
    ) {
        let built = OptionParams::builder()
            .spot(s).strike(k).rate(r).dividend_yield(q).volatility(v).expiry(t).build();
        let contract = OptionParams::builder()
            .spot(s).strike(k).rate(r).dividend_yield(q).expiry(t).build_contract();
        if let Ok(p) = built {
            for kind in [OptionType::Call, OptionType::Put] {
                let _ = bsm::price(&p, kind);
                let _ = greeks::analytic(&p, kind);
                let _ = binomial::price(&p, kind, ExerciseStyle::American, steps.min(40));
                let _ = binomial::price(&p, kind, ExerciseStyle::European, steps);
                let _ = binomial::greeks(&p, kind, ExerciseStyle::American, steps.min(40));
                let cfg = McConfig { paths: 64, seed: 1, antithetic: true };
                let _ = mc::european(&p, kind, cfg);
                let _ = mc::american_lsm(&p, kind, steps.min(8), cfg);
            }
        }
        if let Ok(c) = contract {
            for kind in [OptionType::Call, OptionType::Put] {
                let _ = implied_vol::solve(&c, kind, price);
            }
        }
    }

    #[test]
    fn oversized_simulations_return_errors(
        paths in prop_oneof![Just(usize::MAX), Just(usize::MAX / 2), Just(1usize << 40)],
        steps in prop_oneof![Just(usize::MAX), Just(1usize << 40), Just(1usize << 20)],
    ) {
        let p = OptionParams::builder()
            .spot(100.0).strike(100.0).volatility(0.2).expiry(1.0).build().unwrap();
        let cfg = McConfig { paths, seed: 1, antithetic: false };
        prop_assert!(mc::american_lsm(&p, OptionType::Put, steps, cfg).is_err());
        prop_assert!(binomial::price(&p, OptionType::Put, ExerciseStyle::American, usize::MAX).is_err());
    }

    #[test]
    fn distribution_and_risk_never_panic(
        x in wild(), conf in wild(), rets in wild_vec(40), ppy in wild(), rf in wild(),
    ) {
        let _ = dist::norm_pdf(x);
        let _ = dist::norm_cdf(x);
        for value in [dist::norm_inv_cdf(x), risk::historical_var(&rets, conf),
            risk::historical_cvar(&rets, conf), risk::parametric_var(&rets, conf),
            risk::sharpe_ratio(&rets, rf, ppy), risk::sortino_ratio(&rets, rf, ppy),
            risk::max_drawdown(&rets), risk::annualized_volatility(&rets, ppy)].into_iter().flatten() {
            prop_assert!(value.is_finite());
        }
    }

    #[test]
    fn surface_never_panics(
        spot in wild(), rate in wild(), expiry in wild(),
        strikes in wild_vec(6), vols in wild_vec(6), q_strike in wild(), q_expiry in wild(),
        ks in wild_vec(12), ws in wild_vec(12),
        a in wild(), b in wild(), rho in wild(), m in wild(), sigma in wild(),
    ) {
        let slice = SurfaceSlice { expiry, strikes, vols };
        if let Ok(s) = VolSurface::new(spot, rate, 0.0, vec![slice]) {
            let _ = s.vol(q_strike, q_expiry);
            let _ = s.check_arbitrage();
        }
        let _ = fit_svi(&ks, &ws);
        if let Ok(svi) = SviParams::new(a, b, rho, m, sigma) {
            let _ = svi.total_variance(q_strike);
            let _ = svi.butterfly_g(q_strike);
            let _ = svi.is_butterfly_free(-1.0, 1.0, 11);
        }
    }

    #[test]
    fn batch_never_panics(
        spots in wild_vec(5), strike in wild(), vol in wild(), t in wild(),
    ) {
        let params: Vec<OptionParams> = spots
            .iter()
            .filter_map(|&s| {
                OptionParams::builder().spot(s).strike(strike).volatility(vol).expiry(t).build().ok()
            })
            .collect();
        let _ = batch::price(&params, OptionType::Call);
        let _ = batch::greeks(&params, OptionType::Put);
        let _ = batch::binomial_price(&params, OptionType::Put, ExerciseStyle::American, 10);
        let _ = batch::grid(&spots, &[strike], &[t], &[vol], 0.0, 0.0);
    }
}
