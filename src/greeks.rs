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
use crate::error::{QuantError, Result};
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

/// Smallest step, relative to the parameter's scale, that [`numerical`]
/// accepts. A second difference divides rounding noise of order `eps * price`
/// by the square of the step, so below about `1e-6` of the parameter the
/// results are noise.
const MIN_REL_STEP: f64 = 1e-6;

/// First and second derivative of `f` at `x` from bumps of size about `h`.
///
/// * The steps are the **actual** floating-point differences `up - x` and
///   `x - down`, not the requested `h`, so a bump that is partly or wholly
///   absorbed by rounding is neither silently mis-scaled nor divided by zero:
///   it is rejected if either actual step is not positive or is below
///   [`MIN_REL_STEP`] of `scale`.
/// * Central differences with the (possibly slightly unequal) actual steps are
///   used when both neighbours are valid inputs. If a neighbour falls outside
///   the supported domain (the callback returns an error), the one-sided
///   second-order formulas on the valid side are used instead:
///   `f' = (-3 f0 + 4 f1 - 2 f2 ... )` in the standard three-point form and
///   `f'' = (2 f0 - 5 f1 + 4 f2 - f3) / h^2` (and mirrored for the backward
///   side), which need two or three more evaluations on that side.
/// * A callback value that is not finite is an error, never an input to the
///   differences.
fn derivatives12<F>(f: F, x: f64, h: f64, scale: f64, f0: f64) -> Result<(f64, f64)>
where
    F: Fn(f64) -> Result<f64>,
{
    let not_finite = || QuantError::NonFinite { name: "price" };
    let checked = |v: Result<f64>| -> Result<f64> {
        let v = v?;
        if v.is_finite() {
            Ok(v)
        } else {
            Err(not_finite())
        }
    };
    let (up, down) = (x + h, x - h);
    let (h1, h2) = (up - x, x - down);
    let floor = MIN_REL_STEP * scale;
    if !(h1 > 0.0 && h2 > 0.0) || h1 < floor || h2 < floor || !h.is_finite() {
        return Err(QuantError::InvalidInput {
            name: "bumps",
            reason: "bump is too small to change the input by at least 1e-6 of its scale in double precision, or is not finite",
        });
    }

    let central = (checked(f(up)), checked(f(down)));
    match central {
        (Ok(fu), Ok(fd)) => {
            let d1 = (h2 * h2 * (fu - f0) - h1 * h1 * (fd - f0)) / (h1 * h2 * (h1 + h2));
            let d2 = 2.0 * ((fu - f0) / h1 - (f0 - fd) / h2) / (h1 + h2);
            return finite_pair(d1, d2);
        }
        // A non-finite price is a real failure, not a domain boundary.
        (Err(e @ QuantError::NonFinite { .. }), _) | (_, Err(e @ QuantError::NonFinite { .. })) => {
            return Err(e);
        }
        _ => {}
    }

    // Domain boundary: try the forward, then the backward one-sided formulas.
    let one_sided = |sign: f64| -> Result<(f64, f64)> {
        let x1 = x + sign * h;
        let x2 = x + sign * 2.0 * h;
        let x3 = x + sign * 3.0 * h;
        let step = sign * (x1 - x);
        if step.is_nan() || step < floor || ((x2 - x1) * sign - step).abs() > 1e-6 * step {
            return Err(QuantError::InvalidInput {
                name: "bumps",
                reason: "bump is too small for one-sided differences",
            });
        }
        let (f1, f2, f3) = (checked(f(x1))?, checked(f(x2))?, checked(f(x3))?);
        // For the backward side the odd derivative changes sign.
        let d1 = sign * (-3.0 * f0 + 4.0 * f1 - f2) / (2.0 * step);
        let d2 = (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f3) / (step * step);
        finite_pair(d1, d2)
    };
    match one_sided(1.0) {
        Ok(v) => Ok(v),
        Err(e @ QuantError::NonFinite { .. }) => Err(e),
        Err(_) => one_sided(-1.0).or(Err(QuantError::InvalidInput {
            name: "bumps",
            reason: "input is too close to the supported-domain boundary on both sides for this bump size",
        })),
    }
}

fn finite_pair(d1: f64, d2: f64) -> Result<(f64, f64)> {
    if d1.is_finite() && d2.is_finite() {
        Ok((d1, d2))
    } else {
        Err(QuantError::NonFinite {
            name: "numerical derivative",
        })
    }
}

/// Bump-and-reprice Greeks using central differences.
///
/// `price` is called with bumped copies of `p` and may be any pricer, for
/// example a closure over the binomial tree.
///
/// * Delta, gamma, vega, theta and rho come from central differences that use
///   the **actual** floating-point step sizes, so a rounded bump is neither
///   mis-scaled nor divided by zero.
/// * Near a boundary of the supported domain (see [`crate::limits`]; for
///   example spot near `1e-8`, volatility near its minimum, or a rate at the
///   edge of its range) the neighbour on that side is not a valid input, and
///   the second-order one-sided formulas on the other side are used instead.
/// * Bumps that cannot change the input by at least `1e-6` of its scale
///   (spot, volatility and expiry: their own magnitude; rate: `max(|rate|,
///   0.01)`) are rejected, because second differences of such steps are
///   dominated by rounding.
/// * Volatility and time bumps are shrunk to at most half the parameter so the
///   down-bump stays positive.
///
/// # Errors
///
/// * [`QuantError::InvalidInput`] for non-positive or non-finite bump sizes,
///   bumps too small to be usable, or an input too close to the domain
///   boundary on both sides.
/// * [`QuantError::NonFinite`] if `price` returns a non-finite value, or a
///   difference overflows.
/// * Any error returned by `price` itself, and by re-validating bumped inputs
///   when no one-sided fallback applies.
///
/// # Examples
///
/// ```
/// use quantkit::{bsm, greeks, OptionParams, OptionType};
/// let p = OptionParams::builder().spot(100.0).strike(100.0).rate(0.05).volatility(0.2).expiry(1.0).build()?;
/// let num = greeks::numerical(&p, |q| Ok(bsm::price(q, OptionType::Call)), Default::default())?;
/// let ana = greeks::analytic(&p, OptionType::Call);
/// assert!((num.delta - ana.delta).abs() < 1e-4);
/// // A bump that cannot change the input is an error, not a zero.
/// let tiny = greeks::Bumps { spot_rel: f64::MIN_POSITIVE, ..Default::default() };
/// assert!(greeks::numerical(&p, |q| Ok(bsm::price(q, OptionType::Call)), tiny).is_err());
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
    let eval = |q: Result<OptionParams>| -> Result<f64> { price(&q?) };
    let base = {
        let v = price(p)?;
        if !v.is_finite() {
            return Err(QuantError::NonFinite { name: "price" });
        }
        v
    };

    let (spot, vol, rate, expiry) = (p.spot(), p.volatility(), p.rate(), p.expiry());
    let (delta, gamma) = derivatives12(
        |x| eval(p.with_spot(x)),
        spot,
        spot * bumps.spot_rel,
        spot,
        base,
    )?;
    let (vega, _) = derivatives12(
        |x| eval(p.with_vol(x)),
        vol,
        bumps.vol.min(0.5 * vol),
        vol,
        base,
    )?;
    let (rho, _) = derivatives12(
        |x| eval(p.with_rate(x)),
        rate,
        bumps.rate,
        rate.abs().max(0.01),
        base,
    )?;
    // theta = dV/dt = -dV/dT
    let (dv_dt, _) = derivatives12(
        |x| eval(p.with_expiry(x)),
        expiry,
        bumps.time.min(0.5 * expiry),
        expiry,
        base,
    )?;

    Ok(Greeks {
        delta,
        gamma,
        vega,
        theta: -dv_dt,
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

    fn call_price(x: &OptionParams) -> Result<f64> {
        Ok(bsm::price(x, OptionType::Call))
    }

    /// The reproduction: a bump so small that the bumped spot equals the spot.
    #[test]
    fn bumps_that_cannot_change_the_input_are_rejected() {
        let p = params(100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        for tiny in [f64::MIN_POSITIVE, 1e-300, 1e-20, 1e-17, 1e-9] {
            for bumps in [
                Bumps {
                    spot_rel: tiny,
                    ..Bumps::default()
                },
                Bumps {
                    vol: tiny,
                    ..Bumps::default()
                },
                Bumps {
                    rate: tiny,
                    ..Bumps::default()
                },
                Bumps {
                    time: tiny,
                    ..Bumps::default()
                },
            ] {
                assert!(
                    matches!(
                        numerical(&p, call_price, bumps),
                        Err(QuantError::InvalidInput { .. })
                    ),
                    "bumps {bumps:?} should be rejected"
                );
            }
        }
        // Borderline: 5e-7 of the scale is below the 1e-6 floor, 1e-5 is above it.
        assert!(numerical(
            &p,
            call_price,
            Bumps {
                spot_rel: 5e-7,
                ..Bumps::default()
            }
        )
        .is_err());
        assert!(numerical(
            &p,
            call_price,
            Bumps {
                vol: 1e-7,
                ..Bumps::default()
            }
        )
        .is_err());
        assert!(numerical(
            &p,
            call_price,
            Bumps {
                rate: 1e-8,
                ..Bumps::default()
            }
        )
        .is_err());
        assert!(numerical(
            &p,
            call_price,
            Bumps {
                time: 5e-7,
                ..Bumps::default()
            }
        )
        .is_err());
        assert!(numerical(
            &p,
            call_price,
            Bumps {
                spot_rel: 1e-5,
                ..Bumps::default()
            }
        )
        .is_ok());
        // Just above the accepted floor works and stays accurate enough for what
        // a 1e-6 step can deliver (first derivatives; gamma is noisy but finite).
        let ok = Bumps {
            spot_rel: 1e-3,
            ..Bumps::default()
        };
        assert!(numerical(&p, call_price, ok).unwrap().gamma.is_finite());
    }

    #[test]
    fn non_finite_pricer_values_are_errors() {
        let p = params(100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        let base_call = |x: &OptionParams| Ok(bsm::price(x, OptionType::Call));
        assert!(numerical(&p, base_call, Bumps::default()).is_ok());
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let always = move |_: &OptionParams| Ok(bad);
            assert!(matches!(
                numerical(&p, always, Bumps::default()),
                Err(QuantError::NonFinite { .. })
            ));
            // NaN only on bumped inputs, finite at the base point.
            let only_bumped = move |x: &OptionParams| {
                Ok(
                    if x.spot() == 100.0
                        && x.volatility() == 0.2
                        && x.rate() == 0.05
                        && x.expiry() == 1.0
                    {
                        1.0
                    } else {
                        bad
                    },
                )
            };
            assert!(matches!(
                numerical(&p, only_bumped, Bumps::default()),
                Err(QuantError::NonFinite { .. })
            ));
        }
        // Errors from the pricer are passed through.
        let failing = |_: &OptionParams| -> Result<f64> {
            Err(QuantError::InvalidInput {
                name: "x",
                reason: "boom",
            })
        };
        assert!(matches!(
            numerical(&p, failing, Bumps::default()),
            Err(QuantError::InvalidInput { name: "x", .. })
        ));
    }

    /// At the edges of the supported domain the symmetric neighbours are not
    /// valid inputs; the one-sided second-order formulas must take over and
    /// stay close to the analytic Greeks.
    #[test]
    fn domain_edges_use_one_sided_differences() {
        use crate::limits::*;
        // Spot at the lower edge: down-bump leaves the domain.
        let low = params(MIN_PRICE, 1e-8, 0.05, 0.0, 0.2, 1.0);
        let n = numerical(&low, call_price, Bumps::default()).unwrap();
        let a = analytic(&low, OptionType::Call);
        assert!(
            (n.delta - a.delta).abs() < 2e-3,
            "delta {} vs {}",
            n.delta,
            a.delta
        );
        assert!(
            (n.gamma - a.gamma).abs() <= 0.05 * a.gamma.abs() + 1e-3 * a.delta.abs() / MIN_PRICE,
            "gamma {} vs {}",
            n.gamma,
            a.gamma
        );
        // Spot at the upper edge: up-bump leaves the domain.
        let high = params(MAX_PRICE, MAX_PRICE, 0.05, 0.0, 0.2, 1.0);
        let n = numerical(&high, call_price, Bumps::default()).unwrap();
        let a = analytic(&high, OptionType::Call);
        assert!(
            (n.delta - a.delta).abs() < 2e-3,
            "delta {} vs {}",
            n.delta,
            a.delta
        );
        assert!((n.vega / a.vega - 1.0).abs() < 1e-2);
        // Rate at the upper edge (|rate| * expiry bound irrelevant at T = 1).
        let r_edge = params(100.0, 100.0, MAX_ABS_RATE, 0.0, 0.2, 1.0);
        let n = numerical(&r_edge, call_price, Bumps::default()).unwrap();
        let a = analytic(&r_edge, OptionType::Call);
        assert!(
            (n.rho / a.rho - 1.0).abs() < 1e-3,
            "rho {} vs {}",
            n.rho,
            a.rho
        );
        // Rate bound through |rate| * expiry = 500.
        let rt_edge = params(100.0, 100.0, 5.0, 0.0, 0.4, 100.0);
        let n = numerical(&rt_edge, call_price, Bumps::default()).unwrap();
        assert!(n.rho.is_finite() && n.theta.is_finite());
        // Volatility and expiry at their minimum.
        let tiny = params(100.0, 100.0, 0.05, 0.0, MIN_VOL, MIN_EXPIRY);
        let n = numerical(&tiny, call_price, Bumps::default()).unwrap();
        assert!([n.delta, n.gamma, n.vega, n.theta, n.rho]
            .iter()
            .all(|x| x.is_finite()));
        // Expiry and volatility at their maximum. The default absolute bumps
        // (vol 0.001, time 1/365) are below 1e-6 of a value of 1000, so they are
        // rejected there rather than silently resized; explicit bumps work.
        let big = params(100.0, 100.0, 0.0, 0.0, MAX_VOL, MAX_EXPIRY);
        assert!(numerical(&big, call_price, Bumps::default()).is_err());
        let wide = Bumps {
            vol: 1.0,
            time: 1.0,
            ..Bumps::default()
        };
        let n = numerical(&big, call_price, wide).unwrap();
        assert!([n.delta, n.gamma, n.vega, n.theta, n.rho]
            .iter()
            .all(|x| x.is_finite()));
    }

    /// Accuracy of the actual-step formulas on ordinary inputs is unchanged:
    /// 5e-3 relative bump, 1e-4 for first derivatives, as before.
    #[test]
    fn unequal_actual_steps_do_not_hurt_ordinary_accuracy() {
        // 0.1 + 0.2 style inputs where x + h and x - h do not round symmetrically.
        let p = params(0.1 + 0.2, 0.3, 0.07, 0.01, 0.31, 0.7);
        let n = numerical(&p, call_price, Bumps::default()).unwrap();
        let a = analytic(&p, OptionType::Call);
        assert!((n.delta - a.delta).abs() < 2e-4);
        assert!((n.gamma - a.gamma).abs() < 5e-3 * a.gamma.abs() + 1e-3);
        assert!((n.vega / a.vega - 1.0).abs() < 1e-5);
        assert!((n.rho / a.rho - 1.0).abs() < 1e-5);
        assert!((n.theta - a.theta).abs() < 1e-3 * a.theta.abs() + 1e-3);
    }
}
