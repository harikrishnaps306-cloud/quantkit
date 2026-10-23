//! Implied volatility under Black-Scholes-Merton.
//!
//! # Method
//!
//! The price is split into intrinsic value and time value. The time value,
//! divided by `sqrt(Se^{-qT} Ke^{-rT})`, is the normalised Black function
//! `b(x, s)` of [`crate::black`], with `x = ln(F/K)` and `s = sigma sqrt(T)`.
//! The solver finds `s` from `ln b(-|x|, s) = ln(time value / scale)` with a
//! bracketed Newton iteration that falls back to geometric bisection. Working
//! in logarithms keeps tiny prices (down to the smallest `f64`) from
//! underflowing and makes the iteration scale-free, and `b` is evaluated by a
//! numerically stable method far out of the money, so evaluation does not lose
//! precision in the tails. What a price says about the volatility is a separate
//! matter of conditioning, the elasticity `d ln price / d ln vol`, which
//! is large in the tails (good) and small near the upper bound or for in-the-money
//! prices that are almost all intrinsic value (bad); it is reported in
//! [`IvSolution::elasticity`]. Log-domain arithmetic cannot create information
//! that the `f64` price does not contain.
//! This resembles Jaeckel's "Let's Be Rational" in its variables but is a
//! simpler iteration (no rational starting guess, no third-order step).
//!
//! # What the solver guarantees
//!
//! * **On `Ok`**: the returned volatility lies inside the supported range
//!   and the model price at that volatility is within
//!   `(max(tolerance, 8 eps) + 4 eps + 8 eps (1 + |ln price|)) * market_price`
//!   of the market price (absolute price residual, reported in
//!   [`IvSolution::residual`]); the last term is the rounding noise of evaluating
//!   a tiny price as an exponential and is below `1e-13` for prices above `1e-30`.
//! * **Recovery of the generating volatility** is accurate to a few
//!   `1e-13` for out-of-the-money prices at any depth, but only as accurate as
//!   `residual / vega` where vega is small, which includes in-the-money options
//!   priced almost at intrinsic value. [`IvSolution::vega`] is provided so you
//!   can judge that.
//! * **On `Err`**: no volatility is returned. The cause is one of: price
//!   outside the attainable range ([`QuantError::PriceOutOfBounds`]), or the
//!   iteration budget or floating-point resolution was exhausted before the
//!   tolerance was met ([`QuantError::NoConvergence`], which carries the
//!   iteration count and last residual).

use crate::black::{ln_normalised_otm, ln_normalised_vega};
use crate::bsm;
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
    /// BSM vega at `vol` (`d price / d vol`). Volatility error is roughly
    /// `residual / vega`. It can underflow to `0.0` for prices near the
    /// smallest `f64`; use [`elasticity`](Self::elasticity) for conditioning.
    pub vega: f64,
    /// Conditioning of the inversion: the elasticity `d ln price / d ln vol`,
    /// computed in logarithms so it does not underflow. A relative error `e` in
    /// the price becomes a relative volatility error of about `e / elasticity`.
    /// Large in the tails (a tiny price pins the volatility down), small when the
    /// price is saturated near its upper bound or is almost all intrinsic value.
    pub elasticity: f64,
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

    let t = contract.expiry();
    let sqrt_t = t.sqrt();
    let disc_spot = contract.spot() * libm::exp(-contract.dividend_yield() * t);
    let disc_strike = contract.strike() * libm::exp(-contract.rate() * t);
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

    // Work with the out-of-the-money part of the price (time value), in the
    // normalised variables of `crate::black`: x = ln(F/K), s = sigma sqrt(T),
    // and solve  ln b(-|x|, s) = ln(time value / scale)  for s. Because the
    // equation is in logarithms, the iteration is scale-free and b never
    // underflows. (How much a tiny price says about the volatility is a matter of
    // conditioning, the elasticity reported in `IvSolution`, not of this change of
    // variables.)
    let x = libm::log(contract.spot() / contract.strike())
        + (contract.rate() - contract.dividend_yield()) * t;
    let scale = disc_spot.sqrt() * disc_strike.sqrt();
    let time_value = market_price - lower;
    // Below a few units in the last place of the price, the time value is
    // rounding noise of the input (price and intrinsic value are both rounded).
    let resolution = 8.0 * f64::EPSILON * market_price;
    if time_value < resolution {
        return Err(QuantError::PriceOutOfBounds {
            price: market_price,
            lower: lower + resolution,
            upper: upper.min(p_hi),
        });
    }
    let ln_beta = libm::log(time_value) - libm::log(scale);
    let xo = -x.abs();

    let tol = config.tolerance.max(8.0 * f64::EPSILON);
    let (mut lo, mut hi) = (MIN_VOL * sqrt_t, MAX_VOL * sqrt_t);
    let mut s = initial_guess(xo, ln_beta).clamp(lo, hi);

    let finish = |s: f64, iterations: usize, newton: usize| -> Result<IvSolution> {
        let vol = (s / sqrt_t).clamp(MIN_VOL, MAX_VOL);
        let residual = (model(vol)? - market_price).abs();
        // Prices far below 1 are evaluated as exp(log-scale + ln b) with a
        // large negative argument, whose rounding noise is about
        // eps * |ln price|; allow for it so the check is not stricter than the
        // arithmetic can deliver (negligible for prices above 1e-30).
        let noise = 8.0 * f64::EPSILON * (1.0 + libm::log(market_price).abs());
        // The absolute floor keeps the criterion meaningful for subnormal
        // prices, where `tol * price` would round to zero or to one ulp.
        let allowed =
            ((tol + 4.0 * f64::EPSILON + noise) * market_price).max(4.0 * f64::from_bits(1));
        if residual > allowed {
            return Err(QuantError::NoConvergence {
                iterations,
                residual,
            });
        }
        let s_eff = vol * sqrt_t;
        Ok(IvSolution {
            vol,
            residual,
            iterations,
            newton_steps: newton,
            bisection_steps: iterations - newton,
            vega: scale * libm::exp(ln_normalised_vega(xo, s_eff)) * sqrt_t,
            elasticity: libm::exp(
                libm::log(scale) + ln_normalised_vega(xo, s_eff) + libm::log(s_eff)
                    - libm::log(market_price),
            ),
        })
    };

    let (mut newton_steps, mut bisection_steps) = (0usize, 0usize);
    let mut last = f64::INFINITY;
    for _ in 0..config.max_iterations {
        let ln_b = ln_normalised_otm(xo, s);
        let f = ln_b - ln_beta;
        last = f.abs();
        if last <= tol {
            return finish(s, newton_steps + bisection_steps, newton_steps);
        }
        if f > 0.0 {
            hi = s;
        } else {
            lo = s;
        }

        // Newton on f(s) = ln b(s) - ln beta; f' = (db/ds) / b.
        let slope = libm::exp(ln_normalised_vega(xo, s) - ln_b);
        let candidate = s - f / slope;
        if slope.is_finite() && slope > 0.0 && candidate > lo && candidate < hi {
            s = candidate;
            newton_steps += 1;
        } else {
            s = (lo * hi).sqrt(); // geometric bisection: s spans many decades
            bisection_steps += 1;
        }

        if hi - lo <= 4.0 * f64::EPSILON * hi {
            // The bracket cannot shrink further in double precision.
            let f = ln_normalised_otm(xo, s) - ln_beta;
            if f.abs() <= tol {
                return finish(s, newton_steps + bisection_steps, newton_steps);
            }
            return Err(QuantError::NoConvergence {
                iterations: newton_steps + bisection_steps,
                residual: f.abs() * time_value,
            });
        }
    }
    Err(QuantError::NoConvergence {
        iterations: config.max_iterations,
        residual: last * time_value,
    })
}

/// Starting point for the root search: the at-the-money small-`s` relation
/// `b ~ s / sqrt(2 pi)` combined with the far-tail relation
/// `ln b ~ -x^2 / (2 s^2)`. It only needs to land in the right decade; the
/// bracket and bisection fallback guarantee progress from anywhere.
fn initial_guess(xo: f64, ln_beta: f64) -> f64 {
    let atm = (2.0 * std::f64::consts::PI).sqrt() * libm::exp(ln_beta);
    if xo == 0.0 {
        return atm.max(MIN_VOL);
    }
    let tail = xo.abs() / (-2.0 * ln_beta).max(1e-8).sqrt();
    (atm * atm + tail * tail).sqrt()
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

    /// A price a few ulps above intrinsic value carries no volatility
    /// information: the time value is below the resolution of the input.
    #[test]
    fn in_the_money_time_value_below_price_resolution_is_rejected() {
        let c = contract(100.0, 50.0, 0.0, 0.0, 1.0);
        let intrinsic: f64 = 50.0;
        let just_above = f64::from_bits(intrinsic.to_bits() + 3);
        assert!(just_above > intrinsic);
        assert!(matches!(
            solve(&c, OptionType::Call, just_above),
            Err(QuantError::PriceOutOfBounds { .. })
        ));
        // With a visible time value the solve works but is badly conditioned,
        // which the elasticity reports.
        let p = bsm::price(&c.with_volatility(0.05).unwrap(), OptionType::Call);
        let sol = solve_detailed(&c, OptionType::Call, p, SolverConfig::default());
        if let Ok(sol) = sol {
            assert!(sol.elasticity < 1e-3, "elasticity {}", sol.elasticity);
        }
    }

    /// A call worth the spot to many digits has almost no volatility
    /// information: the solve succeeds (it reprices within tolerance) but the
    /// elasticity says the volatility is poorly determined.
    #[test]
    fn prices_saturated_near_the_upper_bound_report_tiny_elasticity() {
        let c = contract(100.0, 100.0, 0.0, 0.0, 1.0);
        let p = bsm::price(&c.with_volatility(14.0).unwrap(), OptionType::Call);
        let sol = solve_detailed(&c, OptionType::Call, p, SolverConfig::default()).unwrap();
        let back = bsm::price(&c.with_volatility(sol.vol).unwrap(), OptionType::Call);
        assert!((back - p).abs() <= 2e-12 * p);
        assert!(sol.elasticity < 1e-6, "elasticity {}", sol.elasticity);
    }

    /// Subnormal prices: a few significant digits only, so the relative
    /// tolerance is replaced by an absolute floor, but the solve is well
    /// defined because the elasticity is huge in the far tail.
    #[test]
    fn subnormal_prices_solve_with_a_meaningful_residual() {
        // d1 = -38 puts the call price near 1.5e-317, below the smallest normal f64.
        let c = contract(
            100.0,
            100.0 * (0.02_f64 * (38.0 + 0.01)).exp(),
            0.0,
            0.0,
            0.01,
        );
        let vol = 0.2;
        let p = bsm::price(&c.with_volatility(vol).unwrap(), OptionType::Call);
        assert!(
            p > 0.0 && p < f64::MIN_POSITIVE,
            "price {p:e} should be subnormal"
        );
        let sol = solve_detailed(&c, OptionType::Call, p, SolverConfig::default()).unwrap();
        assert!((sol.vol - vol).abs() / vol < 1e-8, "vol {}", sol.vol);
        assert!(sol.elasticity > 100.0);
        assert!(sol.residual <= 8.0 * f64::from_bits(1) + 1e-6 * p);
    }
}
