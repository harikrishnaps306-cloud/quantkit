//! Implied volatility under Black-Scholes-Merton.
//!
//! The solver brackets the root, then runs Newton-Raphson. Whenever a Newton
//! step is unusable (vega is vanishing, or the step would leave the bracket)
//! it falls back to bisection, which is guaranteed to converge. If the
//! tolerance is still not reached within the iteration budget the solver
//! returns [`QuantError::NoConvergence`] rather than a bad number.

use crate::bsm;
use crate::dist::norm_pdf;
use crate::error::{check_finite, QuantError, Result};
use crate::option::{Contract, OptionType};

/// Largest volatility the bracket search will try (100,000%).
const MAX_VOL: f64 = 1_000.0;
/// Smallest volatility the bracket starts at.
const MIN_VOL: f64 = 1e-9;

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
///   no-arbitrage range, so no positive volatility reproduces it.
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
    if market_price <= lower || market_price >= upper {
        return Err(QuantError::PriceOutOfBounds {
            price: market_price,
            lower,
            upper,
        });
    }

    let tol = config.tolerance.max(8.0 * f64::EPSILON) * market_price;
    let model = |vol: f64| -> Result<f64> { Ok(bsm::price(&contract.with_volatility(vol)?, kind)) };
    let sqrt_t = contract.expiry().sqrt();

    // Bracket the root: price is increasing in volatility.
    let mut lo = MIN_VOL;
    let mut hi = 1.0;
    while model(hi)? < market_price {
        lo = hi;
        hi *= 2.0;
        if hi > MAX_VOL {
            return Err(QuantError::NoConvergence {
                iterations: 0,
                residual: market_price - model(MAX_VOL)?,
            });
        }
    }

    // Manaster-Koehler style starting point, clamped into the bracket.
    let fwd_log = (contract.spot() / contract.strike()).ln()
        + (contract.rate() - contract.dividend_yield()) * contract.expiry();
    let mut vol = (2.0 * fwd_log.abs() / contract.expiry())
        .sqrt()
        .clamp(0.1, 5.0);
    vol = vol.clamp(lo, hi);

    let mut residual = f64::INFINITY;
    for iteration in 1..=config.max_iterations {
        let diff = model(vol)? - market_price;
        residual = diff.abs();
        if residual <= tol {
            return Ok(vol);
        }
        if diff > 0.0 {
            hi = vol;
        } else {
            lo = vol;
        }

        // Newton step; fall back to bisection when it is not trustworthy.
        let params = contract.with_volatility(vol)?;
        let d1 = crate::bsm::d1_d2(&params).0;
        let vega = contract.spot()
            * (-contract.dividend_yield() * contract.expiry()).exp()
            * norm_pdf(d1)
            * sqrt_t;
        let newton = vol - diff / vega;
        vol = if vega > 1e-14 && newton.is_finite() && newton > lo && newton < hi {
            newton
        } else {
            0.5 * (lo + hi)
        };

        if hi - lo <= f64::EPSILON * hi {
            let diff = model(vol)? - market_price;
            if diff.abs() <= tol {
                return Ok(vol);
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
