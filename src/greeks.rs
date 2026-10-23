//! Option sensitivities ("Greeks").
//!
//! [`analytic`] gives closed-form BSM Greeks. [`numerical`] bumps the inputs
//! of any pricing function and reprices with central differences, which is
//! how Greeks are obtained for the binomial pricer (see
//! [`crate::binomial::greeks`]).
//!
//! # Units
//!
//! | Greek | Meaning | Unit |
//! |---|---|---|
//! | delta | dV/dS | per 1.0 of spot |
//! | gamma | d2V/dS2 | per 1.0 of spot, squared |
//! | vega | dV/d sigma | per 1.00 (100 vol points) of volatility |
//! | theta | dV/dt | per **year** (divide by 365 for per day) |
//! | rho | dV/dr | per 1.00 (100 percentage points) of the rate |

use crate::bsm::d1_d2;
use crate::dist::{norm_cdf, norm_pdf};
use crate::error::Result;
use crate::option::{OptionParams, OptionType};

/// The five standard Greeks. See the [module docs](self) for units.
///
/// # Examples
///
/// ```
/// use quantkit::{greeks, OptionParams, OptionType};
/// let p = OptionParams::builder().spot(100.0).strike(100.0).volatility(0.2).expiry(1.0).build()?;
/// let g = greeks::analytic(&p, OptionType::Call);
/// assert!(g.delta > 0.0 && g.gamma > 0.0 && g.vega > 0.0);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Greeks {
    /// Sensitivity to the spot price.
    pub delta: f64,
    /// Sensitivity of delta to the spot price.
    pub gamma: f64,
    /// Sensitivity to volatility, per 1.00 of volatility.
    pub vega: f64,
    /// Sensitivity to the passage of time, per year.
    pub theta: f64,
    /// Sensitivity to the risk-free rate, per 1.00 of rate.
    pub rho: f64,
}

/// Closed-form Black-Scholes-Merton Greeks (continuous dividend yield).
///
/// # Examples
///
/// Hull's delta example (S = 49, K = 50, r = 5%, sigma = 20%, T = 20 weeks):
///
/// ```
/// use quantkit::{greeks, OptionParams, OptionType};
/// let p = OptionParams::builder()
///     .spot(49.0).strike(50.0).rate(0.05).volatility(0.2).expiry(20.0 / 52.0)
///     .build()?;
/// let g = greeks::analytic(&p, OptionType::Call);
/// assert!((g.delta - 0.522).abs() < 0.0005);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn analytic(p: &OptionParams, kind: OptionType) -> Greeks {
    let (d1, d2) = d1_d2(p);
    let (s, k, r, q, v, t) = (
        p.spot(),
        p.strike(),
        p.rate(),
        p.dividend_yield(),
        p.volatility(),
        p.expiry(),
    );
    let sqrt_t = t.sqrt();
    let eq = libm::exp(-q * t);
    let er = libm::exp(-r * t);
    let pdf_d1 = norm_pdf(d1);

    let gamma = eq * pdf_d1 / (s * v * sqrt_t);
    let vega = s * eq * pdf_d1 * sqrt_t;
    let decay = -s * eq * pdf_d1 * v / (2.0 * sqrt_t);

    match kind {
        OptionType::Call => Greeks {
            delta: eq * norm_cdf(d1),
            gamma,
            vega,
            theta: decay - r * k * er * norm_cdf(d2) + q * s * eq * norm_cdf(d1),
            rho: k * t * er * norm_cdf(d2),
        },
        OptionType::Put => Greeks {
            delta: -eq * norm_cdf(-d1),
            gamma,
            vega,
            theta: decay + r * k * er * norm_cdf(-d2) - q * s * eq * norm_cdf(-d1),
            rho: -k * t * er * norm_cdf(-d2),
        },
    }
}

/// Bump sizes for [`numerical`].
///
/// # Examples
///
/// ```
/// use quantkit::greeks::Bumps;
/// let b = Bumps::default();
/// assert!(b.spot_rel > 0.0);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bumps {
    /// Spot bump as a fraction of spot (default 0.5%).
    pub spot_rel: f64,
    /// Absolute volatility bump (default 0.001, i.e. 0.1 vol points).
    pub vol: f64,
    /// Absolute rate bump (default 0.0001, i.e. 1 basis point).
    pub rate: f64,
    /// Time bump in years (default 1/365; shrunk if it exceeds half the
    /// remaining life).
    pub time: f64,
}

impl Default for Bumps {
    fn default() -> Self {
        Self {
            spot_rel: 0.005,
            vol: 0.001,
            rate: 1e-4,
            time: 1.0 / 365.0,
        }
    }
}

/// Bump-and-reprice Greeks using central differences.
///
/// `price` is called with bumped copies of `p` and may be any pricer, for
/// example a closure over the binomial tree. Delta, gamma, vega, theta and
/// rho are computed from central differences; volatility and time bumps are
/// shrunk when needed to stay valid.
///
/// # Errors
///
/// Propagates any error from `price`, and from re-validating bumped inputs.
/// Also returns [`crate::QuantError::InvalidInput`] if any bump size is not
/// finite and strictly positive.
///
/// # Examples
///
/// ```
/// use quantkit::{bsm, greeks, OptionParams, OptionType};
/// let p = OptionParams::builder().spot(100.0).strike(100.0).rate(0.05).volatility(0.2).expiry(1.0).build()?;
/// let num = greeks::numerical(&p, |q| Ok(bsm::price(q, OptionType::Call)), Default::default())?;
/// let ana = greeks::analytic(&p, OptionType::Call);
/// assert!((num.delta - ana.delta).abs() < 1e-4);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn numerical<F>(p: &OptionParams, price: F, bumps: Bumps) -> Result<Greeks>
where
    F: Fn(&OptionParams) -> Result<f64>,
{
    for (name, v) in [
        ("bumps.spot_rel", bumps.spot_rel),
        ("bumps.vol", bumps.vol),
        ("bumps.rate", bumps.rate),
        ("bumps.time", bumps.time),
    ] {
        if !(v.is_finite() && v > 0.0) {
            return Err(crate::error::QuantError::InvalidInput {
                name,
                reason: "bump sizes must be finite and strictly positive",
            });
        }
    }
    let base = price(p)?;

    let hs = p.spot() * bumps.spot_rel;
    let up = price(&p.with_spot(p.spot() + hs)?)?;
    let dn = price(&p.with_spot(p.spot() - hs)?)?;
    let delta = (up - dn) / (2.0 * hs);
    let gamma = (up - 2.0 * base + dn) / (hs * hs);

    let hv = bumps.vol.min(0.5 * p.volatility());
    let vega = (price(&p.with_vol(p.volatility() + hv)?)?
        - price(&p.with_vol(p.volatility() - hv)?)?)
        / (2.0 * hv);

    let hr = bumps.rate;
    let rho =
        (price(&p.with_rate(p.rate() + hr)?)? - price(&p.with_rate(p.rate() - hr)?)?) / (2.0 * hr);

    // theta = dV/dt = -dV/dT
    let ht = bumps.time.min(0.5 * p.expiry());
    let theta = -(price(&p.with_expiry(p.expiry() + ht)?)?
        - price(&p.with_expiry(p.expiry() - ht)?)?)
        / (2.0 * ht);

    Ok(Greeks {
        delta,
        gamma,
        vega,
        theta,
        rho,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsm;

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

    /// Hull, *Options, Futures, and Other Derivatives*, Greeks chapter:
    /// S0 = 49, K = 50, r = 5%, sigma = 20%, T = 20 weeks, call.
    /// Published (as reported by a secondary source; the book itself was not
    /// available): delta 0.522, gamma 0.066, vega 12.1, theta -4.31 per year.
    /// Hull's rho figure could not be confirmed, so rho is checked against an
    /// independent Python computation instead.
    #[test]
    fn hull_greeks_example() {
        let p = params(49.0, 50.0, 0.05, 0.0, 0.2, 20.0 / 52.0);
        let g = analytic(&p, OptionType::Call);
        assert!((g.delta - 0.522).abs() < 5e-4);
        assert!((g.gamma - 0.066).abs() < 5e-4);
        assert!((g.theta - -4.31).abs() < 5e-3);
        assert!((g.vega - 12.1).abs() < 0.05);
        assert!((g.rho - 8.906_961_949_608_35).abs() < 1e-9);
    }

    // Independent Python 3 (`math.erfc`) computation, double precision.
    #[test]
    fn matches_independent_reference() {
        let p = params(100.0, 95.0, 0.10, 0.05, 0.20, 0.5);
        let c = analytic(&p, OptionType::Call);
        assert!((c.delta - 0.711_128_312_392_26).abs() < 1e-11);
        assert!((c.gamma - 0.022_839_574_296_269_993).abs() < 1e-12);
        assert!((c.vega - 22.839_574_296_269_998).abs() < 1e-10);
        assert!((c.theta - -7.160_658_069_013_175).abs() < 1e-10);
        assert!((c.rho - 30.741_923_858_602_373).abs() < 1e-10);
        let u = analytic(&p, OptionType::Put);
        assert!((u.delta - -0.264_181_599_636_072_6).abs() < 1e-11);
        assert!((u.theta - -3.000_528_096_398_054).abs() < 1e-10);
        assert!((u.rho - -14.441_473_805_181_54).abs() < 1e-10);
    }

    #[test]
    fn put_call_greek_relations() {
        let p = params(100.0, 95.0, 0.10, 0.05, 0.20, 0.5);
        let c = analytic(&p, OptionType::Call);
        let u = analytic(&p, OptionType::Put);
        assert!((c.delta - u.delta - (-0.05_f64 * 0.5).exp()).abs() < 1e-12);
        assert_eq!(c.gamma, u.gamma);
        assert_eq!(c.vega, u.vega);
    }

    #[test]
    fn numerical_matches_analytic_for_bsm() {
        for &(s, k, r, q, v, t) in &[
            (100.0, 100.0, 0.05, 0.0, 0.2, 1.0),
            (100.0, 95.0, 0.10, 0.05, 0.2, 0.5),
            (42.0, 40.0, 0.10, 0.0, 0.2, 0.5),
        ] {
            let p = params(s, k, r, q, v, t);
            for kind in [OptionType::Call, OptionType::Put] {
                let a = analytic(&p, kind);
                let n = numerical(&p, |x| Ok(bsm::price(x, kind)), Bumps::default()).unwrap();
                assert!((a.delta - n.delta).abs() < 1e-4, "delta");
                assert!((a.gamma - n.gamma).abs() < 1e-4, "gamma");
                assert!((a.vega - n.vega).abs() < 1e-3, "vega");
                assert!((a.theta - n.theta).abs() < 1e-3, "theta");
                assert!((a.rho - n.rho).abs() < 1e-3, "rho");
            }
        }
    }

    #[test]
    fn numerical_rejects_bad_bumps() {
        let p = params(100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        let price = |x: &OptionParams| Ok(bsm::price(x, OptionType::Call));
        for bad in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
            for b in [
                Bumps {
                    spot_rel: bad,
                    ..Bumps::default()
                },
                Bumps {
                    vol: bad,
                    ..Bumps::default()
                },
                Bumps {
                    rate: bad,
                    ..Bumps::default()
                },
                Bumps {
                    time: bad,
                    ..Bumps::default()
                },
            ] {
                assert!(numerical(&p, price, b).is_err());
            }
        }
    }

    #[test]
    fn numerical_handles_tiny_vol_and_expiry() {
        let p = params(100.0, 100.0, 0.05, 0.0, 1e-4, 1e-4);
        assert!(numerical(
            &p,
            |x| Ok(bsm::price(x, OptionType::Call)),
            Bumps::default()
        )
        .is_ok());
    }
}
