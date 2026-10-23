//! Cox-Ross-Rubinstein binomial tree for American and European options.
//!
//! With `dt = T / n`, `u = exp(sigma sqrt(dt))`, `d = 1 / u` and risk-neutral
//! up-probability `p = (exp((r - q) dt) - d) / (u - d)`, option values are
//! rolled back through the tree, taking `max(continuation, exercise)` at each
//! node for American options.
//!
//! The tree is evaluated in `O(n^2)` time and `O(n)` memory (one `Vec`).
//! CRR prices converge to BSM at rate `O(1/n)` but oscillate with the parity
//! of `n`, so do not expect monotone convergence.

use crate::error::{QuantError, Result};
use crate::greeks::{self, Bumps, Greeks};
use crate::option::{ExerciseStyle, OptionParams, OptionType};

/// Prices an option on a CRR tree with `steps` time steps.
///
/// # Errors
///
/// * [`QuantError::InvalidInput`] if `steps == 0`, or if `steps` is so small
///   that the risk-neutral probability leaves `(0, 1)` (the tree would admit
///   arbitrage; use more steps).
///
/// # Examples
///
/// ```
/// use quantkit::{binomial, bsm, ExerciseStyle, OptionParams, OptionType};
///
/// let p = OptionParams::builder()
///     .spot(50.0).strike(50.0).rate(0.10).volatility(0.40).expiry(5.0 / 12.0)
///     .build()?;
/// let american = binomial::price(&p, OptionType::Put, ExerciseStyle::American, 500)?;
/// let european = bsm::price(&p, OptionType::Put);
/// assert!(american >= european);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn price(
    p: &OptionParams,
    kind: OptionType,
    style: ExerciseStyle,
    steps: usize,
) -> Result<f64> {
    if steps == 0 {
        return Err(QuantError::InvalidInput {
            name: "steps",
            reason: "must be at least 1",
        });
    }
    let n = steps;
    let dt = p.expiry() / n as f64;
    let u = (p.volatility() * dt.sqrt()).exp();
    let d = 1.0 / u;
    let growth = ((p.rate() - p.dividend_yield()) * dt).exp();
    let prob = (growth - d) / (u - d);
    if !(prob > 0.0 && prob < 1.0) || !u.is_finite() {
        return Err(QuantError::InvalidInput {
            name: "steps",
            reason:
                "too few steps for these parameters: risk-neutral probability is outside (0, 1)",
        });
    }
    let disc = (-p.rate() * dt).exp();
    let (up_w, down_w) = (disc * prob, disc * (1.0 - prob));
    let u2 = u * u;
    let strike = p.strike();
    let payoff = |s: f64| match kind {
        OptionType::Call => (s - strike).max(0.0),
        OptionType::Put => (strike - s).max(0.0),
    };

    // Terminal layer: node j has j up-moves, spot = S * u^(2j - n).
    let mut values: Vec<f64> = Vec::with_capacity(n + 1);
    let mut s = p.spot() * d.powi(n as i32);
    for _ in 0..=n {
        values.push(payoff(s));
        s *= u2;
    }

    for layer in (0..n).rev() {
        let mut s = p.spot() * d.powi(layer as i32);
        for j in 0..=layer {
            let cont = up_w * values[j + 1] + down_w * values[j];
            values[j] = match style {
                ExerciseStyle::European => cont,
                ExerciseStyle::American => cont.max(payoff(s)),
            };
            s *= u2;
        }
    }
    Ok(values[0])
}

/// Bump-and-reprice Greeks for the binomial pricer, using central
/// differences.
///
/// A CRR price is not smooth in spot at the scale of the node spacing
/// (`about 2 sigma sqrt(T/n)`), so a tiny spot bump gives a noisy gamma. The
/// spot bump is therefore widened to four node spacings (capped at 10% of
/// spot); the other bumps are the [`Bumps`] defaults.
///
/// Useful for comparing against [`greeks::analytic`] for European options.
///
/// # Errors
///
/// As [`price`], for any of the bumped reprices.
///
/// # Examples
///
/// ```
/// use quantkit::{binomial, ExerciseStyle, OptionParams, OptionType};
///
/// let p = OptionParams::builder()
///     .spot(100.0).strike(100.0).rate(0.05).volatility(0.2).expiry(1.0)
///     .build()?;
/// let g = binomial::greeks(&p, OptionType::Put, ExerciseStyle::American, 200)?;
/// assert!(g.delta < 0.0 && g.gamma > 0.0);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn greeks(
    p: &OptionParams,
    kind: OptionType,
    style: ExerciseStyle,
    steps: usize,
) -> Result<Greeks> {
    let spacing = p.volatility() * (p.expiry() / steps.max(1) as f64).sqrt();
    let bumps = Bumps {
        spot_rel: (4.0 * spacing).clamp(Bumps::default().spot_rel, 0.10),
        ..Bumps::default()
    };
    greeks::numerical(p, |q| price(q, kind, style, steps), bumps)
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

    /// Hull, *Options, Futures, and Other Derivatives*, binomial trees
    /// chapter: two-step American put, S0 = 50, K = 52, r = 5%, sigma = 30%,
    /// T = 2 years, gives 7.43 (u = 1.3499, d = 0.7408, p = 0.5097).
    #[test]
    fn hull_two_step_american_put() {
        let p = params(50.0, 52.0, 0.05, 0.0, 0.30, 2.0);
        let v = price(&p, OptionType::Put, ExerciseStyle::American, 2).unwrap();
        assert!((v - 7.43).abs() < 0.005, "v = {v}");
    }

    /// Hull, same chapter's numerical-procedures figure: American put with
    /// S0 = 50, K = 50, r = 10%, sigma = 40%, T = 5 months on a five-step
    /// tree is worth 4.49.
    #[test]
    fn hull_five_step_american_put() {
        let p = params(50.0, 50.0, 0.10, 0.0, 0.40, 5.0 / 12.0);
        let v = price(&p, OptionType::Put, ExerciseStyle::American, 5).unwrap();
        assert!((v - 4.49).abs() < 0.005, "v = {v}");
    }

    #[test]
    fn european_converges_to_bsm() {
        let p = params(100.0, 100.0, 0.05, 0.02, 0.2, 1.0);
        for kind in [OptionType::Call, OptionType::Put] {
            let exact = bsm::price(&p, kind);
            let err = |n| (price(&p, kind, ExerciseStyle::European, n).unwrap() - exact).abs();
            assert!(err(1000) < 5e-3, "{kind:?} 1000: {}", err(1000));
            assert!(err(1000) < err(10));
            assert!(err(4000) < 2e-3, "{kind:?} 4000: {}", err(4000));
        }
    }

    #[test]
    fn american_at_least_european_and_intrinsic() {
        let p = params(100.0, 110.0, 0.06, 0.0, 0.25, 1.0);
        let am = price(&p, OptionType::Put, ExerciseStyle::American, 400).unwrap();
        let eu = price(&p, OptionType::Put, ExerciseStyle::European, 400).unwrap();
        assert!(am >= eu);
        assert!(am >= 10.0 - 1e-12); // K - S
        assert!(
            am > eu + 1e-3,
            "positive rates make early exercise valuable"
        );
    }

    #[test]
    fn american_call_without_dividends_equals_european() {
        let p = params(100.0, 100.0, 0.05, 0.0, 0.3, 1.0);
        let am = price(&p, OptionType::Call, ExerciseStyle::American, 300).unwrap();
        let eu = price(&p, OptionType::Call, ExerciseStyle::European, 300).unwrap();
        assert!((am - eu).abs() < 1e-10);
    }

    #[test]
    fn rejects_bad_steps() {
        let p = params(100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        assert!(price(&p, OptionType::Call, ExerciseStyle::European, 0).is_err());
        // Huge rate, tiny vol, one step: p > 1.
        let bad = params(100.0, 100.0, 0.5, 0.0, 0.01, 1.0);
        assert!(price(&bad, OptionType::Call, ExerciseStyle::European, 1).is_err());
    }

    #[test]
    fn numerical_greeks_match_analytic_for_european() {
        let p = params(100.0, 100.0, 0.05, 0.0, 0.2, 1.0);
        for kind in [OptionType::Call, OptionType::Put] {
            let a = greeks::analytic(&p, kind);
            let n = greeks(&p, kind, ExerciseStyle::European, 2000).unwrap();
            assert!(
                (a.delta - n.delta).abs() < 2e-3,
                "delta {} {}",
                a.delta,
                n.delta
            );
            assert!(
                (a.gamma - n.gamma).abs() < 2e-3,
                "gamma {} {}",
                a.gamma,
                n.gamma
            );
            assert!((a.vega - n.vega).abs() < 0.2, "vega {} {}", a.vega, n.vega);
            assert!(
                (a.theta - n.theta).abs() < 0.1,
                "theta {} {}",
                a.theta,
                n.theta
            );
            assert!((a.rho - n.rho).abs() < 0.2, "rho {} {}", a.rho, n.rho);
        }
    }

    #[test]
    fn extreme_inputs_do_not_panic() {
        let p = params(100.0, 100.0, 0.0, 0.0, 5.0, 1e-6);
        let _ = price(&p, OptionType::Put, ExerciseStyle::American, 50);
        let p = params(100.0, 100.0, 0.0, 0.0, 0.2, 50.0);
        assert!(price(&p, OptionType::Call, ExerciseStyle::American, 100).is_ok());
    }
}
