use quantkit::surface::{SurfaceSlice, VolSurface};
use quantkit::{binomial, dist, risk, ExerciseStyle, OptionParams, OptionType};

#[test]
fn tree_must_not_silently_erase_positive_terminal_nodes() {
    let p = OptionParams::builder()
        .spot(100.0)
        .strike(100.0)
        .rate(0.0)
        .volatility(100.0)
        .expiry(1.0)
        .build()
        .unwrap();
    // Independent log-probability CRR summation gives 99.99999999999507.
    {
        let value = binomial::price(&p, OptionType::Call, ExerciseStyle::European, 100).unwrap();
        assert!((value - 100.0).abs() < 1e-8, "silent wrong price: {value}");
    }
}

#[test]
fn sortino_small_scale_is_not_zero_downside() {
    let xs = [0.02, -0.01, 0.03, -0.02].map(|x| x * 1e-200);
    let value = risk::sortino_ratio(&xs, 0.0, 1.0).unwrap();
    assert!((value - 0.4472135954999579).abs() < 1e-12);
}

#[test]
fn sortino_large_scale_is_not_zero_ratio() {
    let xs = [0.02, -0.01, 0.03, -0.02].map(|x| x * 1e200);
    let value = risk::sortino_ratio(&xs, 0.0, 1.0).unwrap();
    assert!((value - 0.4472135954999579).abs() < 1e-12, "got {value}");
}

#[test]
fn cvar_success_must_be_finite() {
    let xs = [-f64::MAX, f64::MAX, f64::MAX];
    {
        let value = risk::historical_cvar(&xs, 0.1).unwrap();
        assert!((value / f64::MAX + 1.0 / 3.0).abs() < 1e-15);
        assert!(value.is_finite(), "successful CVaR is {value}");
    }
}

#[test]
fn inverse_normal_success_must_be_finite() {
    {
        let value = dist::norm_inv_cdf(f64::from_bits(1)).unwrap();
        // CPython statistics.NormalDist inverse, independently implemented.
        assert!((value + 38.46740561714434).abs() < 2e-13);
        assert!(value.is_finite(), "successful quantile is {value}");
    }
}

#[test]
fn surface_must_reproduce_accepted_quote_nodes() {
    let surface = VolSurface::new(
        100.0,
        10.0,
        -10.0,
        vec![SurfaceSlice {
            expiry: 50.0,
            strikes: vec![90.0, 100.0, 110.0],
            vols: vec![0.2, 0.3, 0.4],
        }],
    );
    {
        let surface = surface.unwrap();
        let value = surface.vol(110.0, 50.0).unwrap();
        assert!(
            (value - 0.4).abs() < 1e-12,
            "accepted quote changed to {value}"
        );
    }
}

#[test]
fn risk_ratios_survive_opposite_sign_maxima() {
    let xs = [-f64::MAX, f64::MAX, f64::MAX];
    let sharpe = risk::sharpe_ratio(&xs, 0.0, 1.0).unwrap();
    assert!((sharpe - 1.0 / (12.0_f64).sqrt()).abs() < 1e-14);
    let sortino = risk::sortino_ratio(&xs, 0.0, 1.0).unwrap();
    assert!((sortino - 1.0 / 3.0_f64.sqrt()).abs() < 1e-14);
    assert!(risk::annualized_volatility(&xs, 1.0).is_err());
}

#[test]
fn normal_quantile_tail_references() {
    // CPython statistics.NormalDist().inv_cdf(p), not an inverse roundtrip.
    for (p, expected) in [
        (1e-300, -37.0470962993612),
        (1e-100, -21.27345356096532),
        (1e-20, -9.262340089798405),
        (1.0 - f64::EPSILON, 8.125890664701906),
    ] {
        let value = dist::norm_inv_cdf(p).unwrap();
        assert!((value - expected).abs() < 2e-13, "{p}: {value}");
    }
}

#[test]
fn extreme_tree_put_call_parity_and_early_exercise() {
    for (vol, steps) in [(60.0, 200), (100.0, 100), (30.0, 500)] {
        let p = OptionParams::builder()
            .spot(120.0)
            .strike(100.0)
            .rate(0.03)
            .dividend_yield(0.01)
            .volatility(vol)
            .expiry(1.0)
            .build()
            .unwrap();
        let c = binomial::price(&p, OptionType::Call, ExerciseStyle::European, steps).unwrap();
        let put = binomial::price(&p, OptionType::Put, ExerciseStyle::European, steps).unwrap();
        let parity = 120.0 * (-0.01_f64).exp() - 100.0 * (-0.03_f64).exp();
        assert!((c - put - parity).abs() < 1e-8);
        let american =
            binomial::price(&p, OptionType::Put, ExerciseStyle::American, steps).unwrap();
        assert!(american + 1e-10 >= put);
        assert!(american <= 100.0 + 1e-8);
    }
}

#[test]
fn invalid_black_arguments_return_promptly() {
    for (x, s) in [
        (1.0, 0.2),
        (-1.0, f64::INFINITY),
        (f64::NEG_INFINITY, 0.2),
        (-1.0, 0.0),
    ] {
        assert!(quantkit::black::normalised_otm(x, s).is_nan());
        assert!(quantkit::black::ln_normalised_otm(x, s).is_nan());
    }
}

#[test]
fn svi_does_not_square_large_finite_coordinates() {
    let svi = quantkit::surface::SviParams::new(0.0, 1e-200, 0.0, 0.0, 1e200).unwrap();
    assert!((svi.total_variance(1e200) - 2.0_f64.sqrt()).abs() < 1e-14);
}

#[test]
fn numerical_greeks_propagate_pricer_errors() {
    let p = OptionParams::builder()
        .spot(100.0)
        .strike(100.0)
        .volatility(0.2)
        .expiry(1.0)
        .build()
        .unwrap();
    let result = quantkit::greeks::numerical(
        &p,
        |q| {
            if q.spot() > 100.0 {
                Err(quantkit::QuantError::NoConvergence {
                    iterations: 7,
                    residual: 0.1,
                })
            } else {
                Ok(quantkit::bsm::price(q, OptionType::Call))
            }
        },
        Default::default(),
    );
    assert!(matches!(
        result,
        Err(quantkit::QuantError::NoConvergence { iterations: 7, .. })
    ));
}
