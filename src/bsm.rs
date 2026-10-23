//! Black-Scholes-Merton prices for European options with a continuous
//! dividend yield.
//!
//! With `d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt(T))` and
//! `d2 = d1 - sigma sqrt(T)`:
//!
//! * call = `S e^{-qT} N(d1) - K e^{-rT} N(d2)`
//! * put = `K e^{-rT} N(-d2) - S e^{-qT} N(-d1)`
//!
//! Inputs are validated when [`OptionParams`] is built (see [`crate::limits`]),
//! so pricing itself is infallible and, within that domain, always returns a
//! finite, non-negative number.

use crate::dist::norm_cdf;
use crate::option::{OptionParams, OptionType};

/// The BSM `d1` and `d2` terms.
pub(crate) fn d1_d2(p: &OptionParams) -> (f64, f64) {
    let sig_sqrt_t = p.volatility() * p.expiry().sqrt();
    let d1 = (libm::log(p.spot() / p.strike())
        + (p.rate() - p.dividend_yield() + 0.5 * p.volatility() * p.volatility()) * p.expiry())
        / sig_sqrt_t;
    (d1, d1 - sig_sqrt_t)
}

/// Black-Scholes-Merton price of a European option.
///
/// # Examples
///
/// Hull's textbook example (S = 42, K = 40, r = 10%, sigma = 20%, T = 0.5):
///
/// ```
/// use quantkit::{bsm, OptionParams, OptionType};
///
/// let p = OptionParams::builder()
///     .spot(42.0).strike(40.0).rate(0.10).volatility(0.20).expiry(0.5)
///     .build()?;
/// assert!((bsm::price(&p, OptionType::Call) - 4.76).abs() < 0.005);
/// assert!((bsm::price(&p, OptionType::Put) - 0.81).abs() < 0.005);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn price(p: &OptionParams, kind: OptionType) -> f64 {
    let (d1, d2) = d1_d2(p);
    let disc_spot = p.spot() * libm::exp(-p.dividend_yield() * p.expiry());
    let disc_strike = p.strike() * libm::exp(-p.rate() * p.expiry());
    // Within the supported domain (see `limits`) the true price is >= 0; the
    // difference of two nearly equal terms can come out as a rounding-sized
    // negative number, which is clamped to zero.
    let raw = match kind {
        OptionType::Call => disc_spot * norm_cdf(d1) - disc_strike * norm_cdf(d2),
        OptionType::Put => disc_strike * norm_cdf(-d2) - disc_spot * norm_cdf(-d1),
    };
    raw.max(0.0)
}

/// Shorthand for `price(p, OptionType::Call)`.
///
/// # Examples
///
/// ```
/// use quantkit::{bsm, OptionParams};
/// let p = OptionParams::builder().spot(100.0).strike(100.0).volatility(0.2).expiry(1.0).build()?;
/// assert!(bsm::call_price(&p) > 0.0);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn call_price(p: &OptionParams) -> f64 {
    price(p, OptionType::Call)
}

/// Shorthand for `price(p, OptionType::Put)`.
///
/// # Examples
///
/// ```
/// use quantkit::{bsm, OptionParams};
/// let p = OptionParams::builder().spot(100.0).strike(100.0).volatility(0.2).expiry(1.0).build()?;
/// assert!(bsm::put_price(&p) > 0.0);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn put_price(p: &OptionParams) -> f64 {
    price(p, OptionType::Put)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Hull, *Options, Futures, and Other Derivatives*, Black-Scholes-Merton
    /// chapter: S0 = 42, K = 40, r = 10%, sigma = 20%, T = 0.5 gives
    /// d1 = 0.7693, d2 = 0.6278, c = 4.76, p = 0.81 (published to 2 dp).
    #[test]
    fn hull_example() {
        let p = params(42.0, 40.0, 0.10, 0.0, 0.20, 0.5);
        let (d1, d2) = d1_d2(&p);
        assert!((d1 - 0.7693).abs() < 5e-5);
        assert!((d2 - 0.6278).abs() < 5e-5);
        assert!((call_price(&p) - 4.76).abs() < 0.005);
        assert!((put_price(&p) - 0.81).abs() < 0.005);
    }

    // The values below were computed with an independent implementation in
    // Python 3 (`math.erfc`, double precision). The ATM case also matches
    // the widely quoted 10.4506 / 5.5735 for S=K=100, r=5%, sigma=20%, T=1.
    type Case = (f64, f64, f64, f64, f64, f64);
    const REFS: [(Case, f64, f64); 4] = [
        (
            (42.0, 40.0, 0.10, 0.0, 0.20, 0.5),
            4.759_422_392_871_528,
            0.808_599_372_900_095_8,
        ),
        (
            (100.0, 100.0, 0.05, 0.0, 0.20, 1.0),
            10.450_583_572_185_565,
            5.573_526_022_256_971,
        ),
        (
            (100.0, 95.0, 0.10, 0.05, 0.20, 0.5),
            9.628_983_522_021_265,
            2.464_787_646_755_823_4,
        ),
        (
            (100.0, 110.0, 0.0, 0.0, 0.30, 2.0),
            13.075_799_854_644_622,
            23.075_799_854_644_62,
        ),
    ];

    #[test]
    fn matches_independent_reference() {
        for ((s, k, r, q, v, t), call, put) in REFS {
            let p = params(s, k, r, q, v, t);
            assert!((call_price(&p) - call).abs() < 1e-11, "call {s} {k}");
            assert!((put_price(&p) - put).abs() < 1e-11, "put {s} {k}");
        }
    }

    #[test]
    fn put_call_parity_with_dividends() {
        let p = params(100.0, 95.0, 0.10, 0.05, 0.20, 0.5);
        let lhs = call_price(&p) - put_price(&p);
        let rhs = 100.0 * (-0.05_f64 * 0.5).exp() - 95.0 * (-0.10_f64 * 0.5).exp();
        assert!((lhs - rhs).abs() < 1e-12);
    }

    #[test]
    fn deep_itm_and_otm() {
        // Deep ITM call is worth its forward intrinsic value.
        let itm = params(200.0, 50.0, 0.05, 0.0, 0.2, 1.0);
        let fwd_intrinsic = 200.0 - 50.0 * (-0.05_f64).exp();
        assert!((call_price(&itm) - fwd_intrinsic).abs() < 1e-9);
        // Deep OTM call is essentially worthless but never negative.
        let otm = params(50.0, 200.0, 0.05, 0.0, 0.2, 1.0);
        let c = call_price(&otm);
        assert!(c > -1e-12 && c < 1e-8, "c={c}");
    }

    #[test]
    fn very_short_expiry_approaches_intrinsic() {
        let call_itm = params(105.0, 100.0, 0.05, 0.0, 0.2, 1e-8);
        assert!((call_price(&call_itm) - 5.0).abs() < 1e-6);
        let put_otm = params(105.0, 100.0, 0.05, 0.0, 0.2, 1e-8);
        assert!(put_price(&put_otm) < 1e-12);
    }

    #[test]
    fn very_high_vol_call_approaches_spot() {
        let p = params(100.0, 100.0, 0.05, 0.0, 50.0, 1.0);
        let c = call_price(&p);
        // d1 = 25, d2 = -25, so the call is worth the whole spot.
        assert!(c.is_finite());
        assert!((c - 100.0).abs() < 1e-9);
    }

    #[test]
    fn zero_rate_atm_call_equals_put() {
        let p = params(100.0, 100.0, 0.0, 0.0, 0.2, 1.0);
        assert!((call_price(&p) - put_price(&p)).abs() < 1e-12);
    }
}
