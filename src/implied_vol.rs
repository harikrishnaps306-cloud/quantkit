//! Implied volatility under Black-Scholes-Merton.
//!
//! # What the solver guarantees
//!
//! The solver brackets the root in `[limits::MIN_VOL, limits::MAX_VOL]`, then
//! runs Newton-Raphson, falling back to bisection whenever a Newton step is
//! unusable (vanishing vega, or a step outside the current bracket).
//!
//! * **On `Ok`**: the returned volatility lies inside the supported range
//!   and the model price at that volatility is within
//!   `max(tolerance, 8 eps) * market_price` of the market price (absolute
//!   price residual, reported in [`IvSolution::residual`]).
//! * **It does not promise** that the volatility is accurate to the same
//!   relative tolerance. Where vega is small (deep in or out of the money,
//!   very short expiry) many volatilities reprice the option equally well;
//!   the volatility error is roughly `residual / vega`, and
//!   [`IvSolution::vega`] is provided so you can judge that.
//! * **On `Err`**: no volatility is returned. The cause is one of: price
//!   outside the attainable range ([`QuantError::PriceOutOfBounds`]), or the
//!   iteration budget or floating-point resolution was exhausted before the
//!   tolerance was met ([`QuantError::NoConvergence`], which carries the
//!   iteration count and last residual).
//!
//! Bisection on a bracket that contains the root halves its width every
//! step, so within the default budget a failure indicates a price that the
//! model cannot resolve in double precision (for example, a call worth
//! `1e-300`), not a slow algorithm.

use crate::bsm;
use crate::dist::norm_pdf;
use crate::error::{check_finite, QuantError, Result};
use crate::option::{Contract, OptionType};

use crate::limits::{MAX_VOL, MIN_VOL};

/// Solver settings.
///
/// # Examples
///
/// ```
/// use quantkit::implied_vol::SolverConfig;
/// let c = SolverConfig::default();
/// assert_eq!(c.max_iterations, 100);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverConfig {
    /// Relative price tolerance: stop when `|model - market| <= tolerance * market`
    /// (never tighter than a few ulps). Default `1e-12`.
    pub tolerance: f64,
    /// Iteration budget. Default 100.
    pub max_iterations: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            max_iterations: 100,
        }
    }
}

/// Solves for the implied volatility with default settings.
///
/// # Errors
///
/// * [`QuantError::NonFinite`] if `market_price` is NaN or infinite.
/// * [`QuantError::PriceOutOfBounds`] if the price is at or outside the
///   attainable range (no-arbitrage bounds, further narrowed to the prices
///   reachable with volatility in `[limits::MIN_VOL, limits::MAX_VOL]`).
/// * [`QuantError::NoConvergence`] if the tolerance is not reached.
///
/// # Examples
///
/// Hull's example: a European call with S = 21, K = 20, r = 10%, T = 0.25
/// trading at 1.875 has an implied volatility of about 23.5%.
///
/// ```
/// use quantkit::{implied_vol, OptionParams, OptionType};
///
/// let c = OptionParams::builder()
///     .spot(21.0).strike(20.0).rate(0.10).expiry(0.25)
///     .build_contract()?;
/// let iv = implied_vol::solve(&c, OptionType::Call, 1.875)?;
/// assert!((iv - 0.235).abs() < 0.001);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn solve(contract: &Contract, kind: OptionType, market_price: f64) -> Result<f64> {
    solve_with(contract, kind, market_price, SolverConfig::default())
}

/// Like [`solve`], with explicit [`SolverConfig`].
///
/// # Errors
///
/// As [`solve`], plus [`QuantError::InvalidInput`] if the configuration has a
/// non-positive tolerance or zero iterations.
///
/// # Examples
///
/// ```
/// use quantkit::implied_vol::{solve_with, SolverConfig};
/// use quantkit::{bsm, OptionParams, OptionType};
///
/// let p = OptionParams::builder().spot(100.0).strike(110.0).rate(0.03).volatility(0.35).expiry(0.75).build()?;
/// let price = bsm::price(&p, OptionType::Put);
/// let cfg = SolverConfig { tolerance: 1e-10, max_iterations: 50 };
/// let iv = solve_with(p.contract(), OptionType::Put, price, cfg)?;
/// assert!((iv - 0.35).abs() < 1e-8);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn solve_with(
    contract: &Contract,
    kind: OptionType,
    market_price: f64,
    config: SolverConfig,
) -> Result<f64> {
    solve_detailed(contract, kind, market_price, config).map(|s| s.vol)
}

/// Diagnostics for a successful solve.
///
/// # Examples
///
/// ```
/// use quantkit::implied_vol::{solve_detailed, SolverConfig};
/// use quantkit::{OptionParams, OptionType};
///
/// let c = OptionParams::builder().spot(21.0).strike(20.0).rate(0.10).expiry(0.25).build_contract()?;
/// let s = solve_detailed(&c, OptionType::Call, 1.875, SolverConfig::default())?;
/// assert!(s.residual < 1e-9 && s.vega > 0.0);
/// assert_eq!(s.iterations, s.newton_steps + s.bisection_steps);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IvSolution {
    /// The implied volatility.
    pub vol: f64,
    /// Absolute difference between the model price at `vol` and the market price.
    pub residual: f64,
    /// Update steps taken before the tolerance was met (`newton_steps + bisection_steps`).
    pub iterations: usize,
    /// How many of those were Newton steps.
    pub newton_steps: usize,
    /// How many were bisection steps (fallback).
    pub bisection_steps: usize,
    /// BSM vega at `vol`. Volatility error is roughly `residual / vega`.
    pub vega: f64,
}

/// Like [`solve_with`], returning [`IvSolution`] diagnostics.
///
/// # Errors
///
/// As [`solve_with`].
///
/// # Examples
///
/// ```
/// use quantkit::implied_vol::{solve_detailed, SolverConfig};
/// use quantkit::{OptionParams, OptionType, QuantError};
///
/// let c = OptionParams::builder().spot(100.0).strike(100.0).rate(0.05).expiry(1.0).build_contract()?;
/// // A call cannot be worth more than the spot.
/// let err = solve_detailed(&c, OptionType::Call, 150.0, SolverConfig::default()).unwrap_err();
/// assert!(matches!(err, QuantError::PriceOutOfBounds { .. }));
/// # Ok::<(), QuantError>(())
/// ```
pub fn solve_detailed(
    contract: &Contract,
    kind: OptionType,
    market_price: f64,
    config: SolverConfig,
) -> Result<IvSolution> {
    check_finite("market_price", market_price)?;
    if !(config.tolerance > 0.0 && config.tolerance.is_finite()) {
        return Err(QuantError::InvalidInput {
            name: "tolerance",
            reason: "must be finite and strictly positive",
        });
    }
    if config.max_iterations == 0 {
        return Err(QuantError::InvalidInput {
            name: "max_iterations",
            reason: "must be at least 1",
        });
    }

    let disc_spot = contract.spot() * (-contract.dividend_yield() * contract.expiry()).exp();
    let disc_strike = contract.strike() * (-contract.rate() * contract.expiry()).exp();
    let (lower, upper) = match kind {
        OptionType::Call => ((disc_spot - disc_strike).max(0.0), disc_spot),
        OptionType::Put => ((disc_strike - disc_spot).max(0.0), disc_strike),
    };
    let model = |vol: f64| -> Result<f64> { Ok(bsm::price(&contract.with_volatility(vol)?, kind)) };

    // The attainable range inside the supported volatility domain.
    let (p_lo, p_hi) = (model(MIN_VOL)?, model(MAX_VOL)?);
    if market_price <= lower || market_price >= upper || market_price < p_lo || market_price > p_hi
    {
        return Err(QuantError::PriceOutOfBounds {
            price: market_price,
            lower: lower.max(p_lo),
            upper: upper.min(p_hi),
        });
    }

    let tol = config.tolerance.max(8.0 * f64::EPSILON) * market_price;
    let sqrt_t = contract.expiry().sqrt();
    let vega_at = |vol: f64| -> Result<f64> {
        let params = contract.with_volatility(vol)?;
        let d1 = crate::bsm::d1_d2(&params).0;
        Ok(contract.spot()
            * (-contract.dividend_yield() * contract.expiry()).exp()
            * norm_pdf(d1)
            * sqrt_t)
    };

    // Bracket: price is increasing in volatility, and p_lo < price < p_hi.
    let (mut lo, mut hi) = (MIN_VOL, MAX_VOL);

    // Manaster-Koehler style starting point, clamped into the bracket.
    let fwd_log = (contract.spot() / contract.strike()).ln()
        + (contract.rate() - contract.dividend_yield()) * contract.expiry();
    let mut vol = (2.0 * fwd_log.abs() / contract.expiry())
        .sqrt()
        .clamp(0.1, 5.0)
        .clamp(lo, hi);

    let (mut newton_steps, mut bisection_steps) = (0usize, 0usize);
    let mut residual = f64::INFINITY;
    for iteration in 1..=config.max_iterations {
        let diff = model(vol)? - market_price;
        residual = diff.abs();
        if residual <= tol {
            return Ok(IvSolution {
                vol,
                residual,
                iterations: newton_steps + bisection_steps,
                newton_steps,
                bisection_steps,
                vega: vega_at(vol)?,
            });
        }
        if diff > 0.0 {
            hi = vol;
        } else {
            lo = vol;
        }

        // Newton step; fall back to bisection when it is not trustworthy.
        let vega = vega_at(vol)?;
        let newton = vol - diff / vega;
        if vega > 1e-14 && newton.is_finite() && newton > lo && newton < hi {
            vol = newton;
            newton_steps += 1;
        } else {
            vol = 0.5 * (lo + hi);
            bisection_steps += 1;
        }

        if hi - lo <= f64::EPSILON * hi {
            // The bracket cannot shrink further in double precision.
            let diff = model(vol)? - market_price;
            if diff.abs() <= tol {
                return Ok(IvSolution {
                    vol,
                    residual: diff.abs(),
                    iterations: newton_steps + bisection_steps,
                    newton_steps,
                    bisection_steps,
                    vega: vega_at(vol)?,
                });
            }
            return Err(QuantError::NoConvergence {
                iterations: iteration,
                residual: diff.abs(),
            });
        }
    }
    Err(QuantError::NoConvergence {
        iterations: config.max_iterations,
        residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::option::OptionParams;

    fn contract(s: f64, k: f64, r: f64, q: f64, t: f64) -> Contract {
        OptionParams::builder()
            .spot(s)
            .strike(k)
            .rate(r)
            .dividend_yield(q)
            .expiry(t)
            .build_contract()
            .unwrap()
    }

    /// Hull, *Options, Futures, and Other Derivatives*, implied volatility
    /// section: S0 = 21, K = 20, r = 10%, T = 3 months, c = 1.875 gives an
    /// implied volatility of 0.235 (confirmed via a secondary source quoting
    /// the book).
    #[test]
    fn hull_implied_vol_example() {
        let c = contract(21.0, 20.0, 0.10, 0.0, 0.25);
        let iv = solve(&c, OptionType::Call, 1.875).unwrap();
        assert!((iv - 0.235).abs() < 5e-4, "iv = {iv}");
        // And the solved vol really reproduces the price.
        let back = bsm::price(&c.with_volatility(iv).unwrap(), OptionType::Call);
        assert!((back - 1.875).abs() < 1e-10);
    }

    #[test]
    fn round_trips_across_regimes() {
        let cases = [
            (100.0, 100.0, 0.05, 0.0, 1.0, 0.2),
            (100.0, 150.0, 0.01, 0.02, 0.1, 0.4),         // OTM
            (100.0, 60.0, 0.01, 0.0, 2.0, 0.3),           // deep ITM call
            (100.0, 100.0, 0.0, 0.0, 0.5, 0.01),          // very low vol
            (100.0, 100.0, 0.02, 0.0, 0.5, 5.0),          // very high vol
            (100.0, 100.0, 0.05, 0.0, 1.0 / 365.0, 0.25), // one day
        ];
        for (s, k, r, q, t, sigma) in cases {
            let c = contract(s, k, r, q, t);
            for kind in [OptionType::Call, OptionType::Put] {
                let price = bsm::price(&c.with_volatility(sigma).unwrap(), kind);
                let iv = solve(&c, kind, price);
                match iv {
                    Ok(iv) => {
                        let back = bsm::price(&c.with_volatility(iv).unwrap(), kind);
                        assert!(
                            (back - price).abs() <= 1e-9 * price.max(1e-12) + 1e-12,
                            "{s} {k} {kind:?}: iv {iv} vs {sigma}"
                        );
                    }
                    // A price that rounds to the bound has no solution.
                    Err(QuantError::PriceOutOfBounds { .. }) => {}
                    Err(e) => panic!("{s} {k} {kind:?}: {e}"),
                }
            }
        }
    }

    #[test]
    fn rejects_prices_outside_bounds() {
        let c = contract(100.0, 100.0, 0.05, 0.0, 1.0);
        assert!(matches!(
            solve(&c, OptionType::Call, 0.0),
            Err(QuantError::PriceOutOfBounds { .. })
        ));
        assert!(matches!(
            solve(&c, OptionType::Call, 100.0),
            Err(QuantError::PriceOutOfBounds { .. })
        ));
        // Below intrinsic: call < S - K e^{-rT} = 4.877...
        let itm = contract(100.0, 90.0, 0.05, 0.0, 1.0);
        assert!(matches!(
            solve(&itm, OptionType::Call, 10.0),
            Err(QuantError::PriceOutOfBounds { .. })
        ));
    }

    #[test]
    fn rejects_nan_and_bad_config() {
        let c = contract(100.0, 100.0, 0.05, 0.0, 1.0);
        assert!(matches!(
            solve(&c, OptionType::Call, f64::NAN),
            Err(QuantError::NonFinite { .. })
        ));
        let bad = SolverConfig {
            tolerance: 0.0,
            max_iterations: 10,
        };
        assert!(solve_with(&c, OptionType::Call, 10.0, bad).is_err());
        let bad = SolverConfig {
            tolerance: 1e-8,
            max_iterations: 0,
        };
        assert!(solve_with(&c, OptionType::Call, 10.0, bad).is_err());
    }

    #[test]
    fn reports_non_convergence_when_budget_too_small() {
        let c = contract(100.0, 100.0, 0.05, 0.0, 1.0);
        let cfg = SolverConfig {
            tolerance: 1e-15,
            max_iterations: 1,
        };
        let r = solve_with(&c, OptionType::Call, 10.123_456, cfg);
        assert!(matches!(r, Err(QuantError::NoConvergence { .. })));
    }
}
