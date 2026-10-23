//! Contract tests for the supported input domain (`quantkit::limits`):
//! every accepted input must produce finite, financially sensible closed-form
//! results, and every input just outside must be rejected.

use quantkit::limits::*;
use quantkit::{bsm, greeks, OptionParams, OptionType};

/// Boundary and interior values of every parameter.
fn spots() -> Vec<f64> {
    vec![MIN_PRICE, 1e-4, 1.0, 100.0, 1e6, MAX_PRICE]
}
fn vols() -> Vec<f64> {
    vec![MIN_VOL, 1e-4, 0.2, 5.0, MAX_VOL]
}
fn expiries() -> Vec<f64> {
    vec![MIN_EXPIRY, 1e-4, 1.0, 100.0, MAX_EXPIRY]
}
fn rates() -> Vec<f64> {
    vec![-MAX_ABS_RATE, -0.5, 0.0, 0.05, MAX_ABS_RATE]
}

#[test]
fn accepted_inputs_give_finite_sensible_results() {
    let mut checked = 0usize;
    for &s in &spots() {
        for &k in &spots() {
            for &v in &vols() {
                for &t in &expiries() {
                    for &r in &rates() {
                        for &q in &[-0.05, 0.0, 0.03, MAX_ABS_RATE] {
                            let Ok(p) = OptionParams::builder()
                                .spot(s)
                                .strike(k)
                                .rate(r)
                                .dividend_yield(q)
                                .volatility(v)
                                .expiry(t)
                                .build()
                            else {
                                // Only the |rate| * expiry bound may reject in-range values.
                                assert!(
                                    r.abs() * t > MAX_RATE_TIME || q.abs() * t > MAX_RATE_TIME,
                                    "unexpected rejection s={s} k={k} v={v} t={t} r={r} q={q}"
                                );
                                continue;
                            };
                            checked += 1;
                            let ctx = format!("s={s:e} k={k:e} v={v:e} t={t:e} r={r} q={q}");
                            let call = bsm::call_price(&p);
                            let put = bsm::put_price(&p);
                            assert!(call.is_finite() && call >= 0.0, "call {call} at {ctx}");
                            assert!(put.is_finite() && put >= 0.0, "put {put} at {ctx}");
                            // No-arbitrage upper bounds.
                            let ds = s * (-q * t).exp();
                            let dk = k * (-r * t).exp();
                            assert!(call <= ds * (1.0 + 1e-12), "call above bound at {ctx}");
                            assert!(put <= dk * (1.0 + 1e-12), "put above bound at {ctx}");
                            // Put-call parity, scaled by the size of the terms.
                            let scale = ds + dk;
                            assert!(
                                ((call - put) - (ds - dk)).abs() <= 1e-9 * scale,
                                "parity at {ctx}: {} vs {}",
                                call - put,
                                ds - dk
                            );
                            for kind in [OptionType::Call, OptionType::Put] {
                                let g = greeks::analytic(&p, kind);
                                for (name, x) in [
                                    ("delta", g.delta),
                                    ("gamma", g.gamma),
                                    ("vega", g.vega),
                                    ("theta", g.theta),
                                    ("rho", g.rho),
                                ] {
                                    assert!(x.is_finite(), "{name} = {x} at {ctx} ({kind:?})");
                                }
                                assert!(g.gamma >= 0.0 && g.vega >= 0.0, "signs at {ctx}");
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 10_000, "grid unexpectedly small: {checked}");
}

#[test]
fn values_just_outside_the_domain_are_rejected() {
    let ok = || {
        OptionParams::builder()
            .spot(100.0)
            .strike(100.0)
            .rate(0.05)
            .volatility(0.2)
            .expiry(1.0)
    };
    assert!(ok().build().is_ok());
    assert!(ok().spot(MIN_PRICE / 2.0).build().is_err());
    assert!(ok().spot(MAX_PRICE * 2.0).build().is_err());
    assert!(ok().strike(MIN_PRICE / 2.0).build().is_err());
    assert!(ok().volatility(MIN_VOL / 2.0).build().is_err());
    assert!(ok().volatility(MAX_VOL * 2.0).build().is_err());
    assert!(ok().expiry(MIN_EXPIRY / 2.0).build().is_err());
    assert!(ok().expiry(MAX_EXPIRY * 2.0).build().is_err());
    assert!(ok().rate(MAX_ABS_RATE * 1.01).build().is_err());
    assert!(ok().rate(-MAX_ABS_RATE * 1.01).build().is_err());
    // |rate| * expiry above 500 would overflow exp().
    assert!(ok().rate(5.0).expiry(200.0).build().is_err());
    assert!(ok().dividend_yield(5.0).expiry(200.0).build().is_err());
    // Re-validation applies to the with_* helpers too.
    let p = ok().build().unwrap();
    assert!(p.with_rate(5.0).and_then(|q| q.with_expiry(200.0)).is_err());
    assert!(p.with_spot(MAX_PRICE * 2.0).is_err());
}
