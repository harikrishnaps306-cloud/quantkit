//! Risk and performance metrics on a slice of periodic returns.
//!
//! # Conventions
//!
//! * Returns are **simple periodic returns as fractions**: `-0.02` is a 2% loss.
//! * VaR and CVaR are reported as **positive loss fractions**. A VaR of `0.03`
//!   at 95% means: in 95% of periods the loss is no worse than 3%. A negative
//!   value means even the tail returns were gains.
//! * Standard deviations use the **sample** (`n - 1`) estimator.
//! * `periods_per_year` annualises: 252 for daily, 52 weekly, 12 monthly.
//! * Every function rejects an empty slice, and any NaN or infinite value.
//! * Successful results are always finite; a result that would overflow is an
//!   error ([`QuantError::NonFinite`]), never `inf` or `NaN`.
//!
//! # Definitions
//!
//! * **Returns** are *simple* period returns `P_t / P_{t-1} - 1`, not log
//!   returns. Convert log returns with `exp(x) - 1` first. A return of `-1`
//!   (total loss) is allowed; below `-1` is impossible and rejected by
//!   [`max_drawdown`].
//! * **Historical VaR** at confidence `c` with `n` observations uses
//!   `k = ceil(n (1 - c))` (at least 1): it is the `k`-th worst return,
//!   negated, with no interpolation. Fractional tail mass therefore rounds
//!   the tail *up* (a slightly larger tail, a slightly more conservative
//!   VaR). Tied returns are fine. With few observations the tail is a single
//!   point, so treat results from `n < 1 / (1 - c)` as the sample minimum.
//! * **Historical CVaR / expected shortfall** is the mean of the same `k`
//!   worst returns, negated; it is always at least the VaR.
//! * **Parametric VaR** assumes normal returns: `z_c * s - m` with the sample
//!   mean `m` and sample standard deviation `s` (`n - 1`).
//! * **Annualisation** multiplies a per-period mean by `periods_per_year` or a
//!   per-period standard deviation by `sqrt(periods_per_year)`. That is only
//!   exact for independent, identically distributed returns.
//! * **Risk-free rate and Sortino target** are *per period* simple returns.
//!   Convert an annual rate `R` yourself, for example `R / 252` (simple) or
//!   `(1 + R)^(1/252) - 1` (compounded); quantkit does not choose for you.
//! * **Max drawdown** is the largest peak-to-trough fall of the wealth path
//!   `W_0 = 1, W_t = W_{t-1} (1 + r_t)`, as a fraction of the running peak.
//!   The initial wealth counts as a peak, so a first-period loss is a drawdown.
//! * **Undefined ratios** (zero volatility for Sharpe, no return below the
//!   target for Sortino) are errors, not infinities.

use crate::dist::norm_inv_cdf;
use crate::error::{check_positive, QuantError, Result};

fn validate(name: &'static str, returns: &[f64]) -> Result<()> {
    if returns.is_empty() {
        return Err(QuantError::EmptyInput { name });
    }
    if returns.iter().any(|r| !r.is_finite()) {
        return Err(QuantError::NonFinite { name });
    }
    Ok(())
}

fn validate_confidence(confidence: f64) -> Result<()> {
    if confidence.is_nan() {
        return Err(QuantError::NonFinite { name: "confidence" });
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(QuantError::InvalidInput {
            name: "confidence",
            reason: "must lie strictly between 0 and 1",
        });
    }
    Ok(())
}

/// Running mean, which cannot overflow when the plain sum would.
fn mean(xs: &[f64]) -> f64 {
    let mut m = 0.0;
    for (i, &x) in xs.iter().enumerate() {
        m += (x - m) / (i + 1) as f64;
    }
    m
}

/// Returns `x` if finite, otherwise an error: successful results are always finite.
fn finite(x: f64) -> Result<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(QuantError::NonFinite { name: "result" })
    }
}

fn sample_std(xs: &[f64]) -> Result<f64> {
    if xs.len() < 2 {
        return Err(QuantError::InvalidInput {
            name: "returns",
            reason: "need at least two observations",
        });
    }
    // Scale by the largest deviation so tiny (1e-200) or huge (1e150) returns
    // neither underflow to a spurious zero variance nor overflow.
    let m = mean(xs);
    let scale = xs.iter().fold(0.0_f64, |a, x| a.max((x - m).abs()));
    if scale == 0.0 || !scale.is_finite() {
        return Ok(scale);
    }
    let ss: f64 = xs.iter().map(|x| ((x - m) / scale).powi(2)).sum();
    Ok(scale * (ss / (xs.len() - 1) as f64).sqrt())
}

/// Number of observations in the tail: `ceil(n * (1 - confidence))`, at
/// least one. The small epsilon stops `100 * (1 - 0.95)` from rounding up.
fn tail_count(n: usize, confidence: f64) -> usize {
    let k = (n as f64 * (1.0 - confidence) - 1e-9).ceil();
    (k as usize).clamp(1, n)
}

/// Historical value at risk: the empirical loss quantile, with no
/// interpolation.
///
/// Takes the `k`-th worst return where `k = ceil(n * (1 - confidence))`
/// and returns its negation.
///
/// # Errors
///
/// Empty or non-finite `returns`, or `confidence` outside `(0, 1)`.
///
/// # Examples
///
/// ```
/// use quantkit::risk;
/// // Twenty returns from -10% to +9% in 1% steps.
/// let r: Vec<f64> = (-10..10).map(|i| f64::from(i) / 100.0).collect();
/// // 95%: the single worst of 20 observations.
/// assert!((risk::historical_var(&r, 0.95)? - 0.10).abs() < 1e-12);
/// // 90%: the second worst.
/// assert!((risk::historical_var(&r, 0.90)? - 0.09).abs() < 1e-12);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn historical_var(returns: &[f64], confidence: f64) -> Result<f64> {
    validate("returns", returns)?;
    validate_confidence(confidence)?;
    let k = tail_count(returns.len(), confidence);
    let mut v = returns.to_vec();
    let (_, kth, _) = v.select_nth_unstable_by(k - 1, f64::total_cmp);
    Ok(-*kth)
}

/// Historical conditional VaR (expected shortfall): the average loss over
/// the `k` worst returns, with `k` as in [`historical_var`]. Always at least
/// as large as the VaR at the same confidence.
///
/// # Errors
///
/// Empty or non-finite `returns`, or `confidence` outside `(0, 1)`.
///
/// # Examples
///
/// ```
/// use quantkit::risk;
/// let r: Vec<f64> = (-10..10).map(|i| f64::from(i) / 100.0).collect();
/// // 90%: mean of the two worst, -10% and -9%.
/// assert!((risk::historical_cvar(&r, 0.90)? - 0.095).abs() < 1e-12);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn historical_cvar(returns: &[f64], confidence: f64) -> Result<f64> {
    validate("returns", returns)?;
    validate_confidence(confidence)?;
    let k = tail_count(returns.len(), confidence);
    let mut v = returns.to_vec();
    v.select_nth_unstable_by(k - 1, f64::total_cmp);
    Ok(-mean(&v[..k]))
}

/// Parametric (variance-covariance) VaR assuming normally distributed
/// returns: `z * s - m` with `z = Phi^-1(confidence)`, sample mean `m` and
/// sample standard deviation `s`.
///
/// # Errors
///
/// Empty or non-finite `returns`, fewer than two observations, or
/// `confidence` outside `(0, 1)`.
///
/// # Examples
///
/// ```
/// use quantkit::risk;
/// let r = [-0.01, 0.01, -0.01, 0.01];
/// let var = risk::parametric_var(&r, 0.99)?;
/// assert!(var > 0.02 && var < 0.03);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn parametric_var(returns: &[f64], confidence: f64) -> Result<f64> {
    validate("returns", returns)?;
    validate_confidence(confidence)?;
    let s = sample_std(returns)?;
    let z = norm_inv_cdf(confidence)?;
    finite(z * s - mean(returns))
}

/// Annualised Sharpe ratio: `(mean - risk_free) / std * sqrt(periods_per_year)`.
///
/// `risk_free_per_period` is the risk-free return **per period** (for
/// daily data, the annual rate divided by 252, say).
///
/// # Errors
///
/// Empty or non-finite input, fewer than two observations, non-positive
/// `periods_per_year`, or zero volatility (the ratio is undefined).
///
/// # Examples
///
/// ```
/// use quantkit::risk;
/// let r = [0.01, 0.02, -0.01, 0.03];
/// let s = risk::sharpe_ratio(&r, 0.0, 12.0)?;
/// assert!(s > 0.0);
/// assert!(risk::sharpe_ratio(&[0.01, 0.01], 0.0, 12.0).is_err()); // no variance
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn sharpe_ratio(
    returns: &[f64],
    risk_free_per_period: f64,
    periods_per_year: f64,
) -> Result<f64> {
    validate("returns", returns)?;
    crate::error::check_finite("risk_free_per_period", risk_free_per_period)?;
    check_positive("periods_per_year", periods_per_year)?;
    let s = sample_std(returns)?;
    if s == 0.0 {
        return Err(QuantError::InvalidInput {
            name: "returns",
            reason: "zero volatility: Sharpe ratio is undefined",
        });
    }
    finite((mean(returns) - risk_free_per_period) / s * periods_per_year.sqrt())
}

/// Annualised Sortino ratio: `(mean - target) / downside_deviation *
/// sqrt(periods_per_year)`, where the downside deviation is
/// `sqrt(mean(min(r - target, 0)^2))` over **all** observations.
///
/// `target_per_period` is the minimum acceptable return per period
/// (often `0.0`).
///
/// # Errors
///
/// Empty or non-finite input, non-positive `periods_per_year`, or no
/// observation below the target (the ratio is undefined).
///
/// # Examples
///
/// ```
/// use quantkit::risk;
/// let r = [0.02, -0.01, 0.03, -0.02];
/// // mean 0.005, downside deviation sqrt((0.01^2 + 0.02^2) / 4)
/// let s = risk::sortino_ratio(&r, 0.0, 1.0)?;
/// assert!((s - 0.447_213_595_499_957_9).abs() < 1e-12);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn sortino_ratio(
    returns: &[f64],
    target_per_period: f64,
    periods_per_year: f64,
) -> Result<f64> {
    validate("returns", returns)?;
    crate::error::check_finite("target_per_period", target_per_period)?;
    check_positive("periods_per_year", periods_per_year)?;
    let worst_shortfall = returns
        .iter()
        .fold(0.0_f64, |a, r| a.max((target_per_period - r).max(0.0)));
    let downside_sq = if worst_shortfall == 0.0 || !worst_shortfall.is_finite() {
        worst_shortfall
    } else {
        let scaled: f64 = returns
            .iter()
            .map(|r| ((r - target_per_period).min(0.0) / worst_shortfall).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        worst_shortfall * worst_shortfall * scaled
    };
    if downside_sq == 0.0 {
        return Err(QuantError::InvalidInput {
            name: "returns",
            reason: "no returns below target: Sortino ratio is undefined",
        });
    }
    finite((mean(returns) - target_per_period) / downside_sq.sqrt() * periods_per_year.sqrt())
}

/// Maximum drawdown of the equity curve implied by compounding `returns`
/// from an initial value of 1: the largest peak-to-trough decline, as a
/// fraction in `[0, 1]`.
///
/// # Errors
///
/// Empty or non-finite `returns`, or any return below `-1` (a loss of more
/// than 100%).
///
/// # Examples
///
/// ```
/// use quantkit::risk;
/// // Equity: 1.10, 0.88, 0.924, 0.8316. Peak 1.10 to trough 0.8316.
/// let dd = risk::max_drawdown(&[0.10, -0.20, 0.05, -0.10])?;
/// assert!((dd - 0.244).abs() < 1e-12);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn max_drawdown(returns: &[f64]) -> Result<f64> {
    validate("returns", returns)?;
    if returns.iter().any(|&r| r < -1.0) {
        return Err(QuantError::InvalidInput {
            name: "returns",
            reason: "a return below -100% is impossible",
        });
    }
    let (mut equity, mut peak, mut worst) = (1.0_f64, 1.0_f64, 0.0_f64);
    for &r in returns {
        equity *= 1.0 + r;
        if !equity.is_finite() {
            return Err(QuantError::NonFinite { name: "equity" });
        }
        peak = peak.max(equity);
        worst = worst.max(1.0 - equity / peak);
    }
    Ok(worst)
}

/// Annualised volatility: sample standard deviation times
/// `sqrt(periods_per_year)`.
///
/// # Errors
///
/// Empty or non-finite `returns`, fewer than two observations, or
/// non-positive `periods_per_year`.
///
/// # Examples
///
/// ```
/// use quantkit::risk;
/// let vol = risk::annualized_volatility(&[0.01, -0.01, 0.01, -0.01], 252.0)?;
/// assert!(vol > 0.0);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn annualized_volatility(returns: &[f64], periods_per_year: f64) -> Result<f64> {
    validate("returns", returns)?;
    check_positive("periods_per_year", periods_per_year)?;
    finite(sample_std(returns)? * periods_per_year.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ladder() -> Vec<f64> {
        (-10..10).map(|i| f64::from(i) / 100.0).collect()
    }

    #[test]
    fn historical_var_cvar_hand_computed() {
        let r = ladder();
        assert!((historical_var(&r, 0.95).unwrap() - 0.10).abs() < 1e-12);
        assert!((historical_cvar(&r, 0.95).unwrap() - 0.10).abs() < 1e-12);
        assert!((historical_var(&r, 0.90).unwrap() - 0.09).abs() < 1e-12);
        assert!((historical_cvar(&r, 0.90).unwrap() - 0.095).abs() < 1e-12);
        // 80%: four worst, -10..-7 percent, mean 8.5%.
        assert!((historical_cvar(&r, 0.80).unwrap() - 0.085).abs() < 1e-12);
    }

    #[test]
    fn order_of_input_is_irrelevant() {
        let mut r = ladder();
        let a = historical_cvar(&r, 0.9).unwrap();
        r.reverse();
        r.swap(3, 11);
        assert_eq!(a, historical_cvar(&r, 0.9).unwrap());
    }

    #[test]
    fn cvar_at_least_var() {
        let r = [
            0.01, -0.03, 0.02, -0.05, 0.00, -0.01, 0.04, -0.02, 0.03, -0.04,
        ];
        for c in [0.5, 0.8, 0.9, 0.99] {
            assert!(historical_cvar(&r, c).unwrap() >= historical_var(&r, c).unwrap());
        }
    }

    #[test]
    fn parametric_var_matches_formula() {
        // mean 0, sample std = sqrt(4 * 1e-4 / 3) = 0.011547005383792516
        let r = [-0.01, 0.01, -0.01, 0.01];
        let v = parametric_var(&r, 0.95).unwrap();
        // z_0.95 = 1.6448536269514715 (statistics.NormalDist().inv_cdf(0.95))
        assert!((v - 1.644_853_626_951_471_5 * 0.011_547_005_383_792_516).abs() < 1e-12);
    }

    #[test]
    fn sharpe_hand_computed() {
        // mean 0.0125; sum sq dev 8.75e-4; sample var 2.91667e-4; std 0.0170783
        let r = [0.01, 0.02, -0.01, 0.03];
        let std = (8.75e-4_f64 / 3.0).sqrt();
        let want = 0.0125 / std * 12.0_f64.sqrt();
        assert!((sharpe_ratio(&r, 0.0, 12.0).unwrap() - want).abs() < 1e-12);
        let with_rf = (0.0125 - 0.001) / std;
        assert!((sharpe_ratio(&r, 0.001, 1.0).unwrap() - with_rf).abs() < 1e-12);
    }

    #[test]
    fn sortino_hand_computed() {
        let r = [0.02, -0.01, 0.03, -0.02];
        // downside deviation = sqrt((1e-4 + 4e-4) / 4); sortino = 0.005 / that.
        let want = 0.005 / (5e-4_f64 / 4.0).sqrt();
        assert!((sortino_ratio(&r, 0.0, 1.0).unwrap() - want).abs() < 1e-12);
        assert!(sortino_ratio(&[0.01, 0.02], 0.0, 1.0).is_err());
    }

    #[test]
    fn drawdown_hand_computed() {
        assert!((max_drawdown(&[0.10, -0.20, 0.05, -0.10]).unwrap() - 0.244).abs() < 1e-12);
        assert_eq!(max_drawdown(&[0.01, 0.02, 0.03]).unwrap(), 0.0);
        assert_eq!(max_drawdown(&[-1.0, 0.5]).unwrap(), 1.0);
        assert!(max_drawdown(&[-1.5]).is_err());
    }

    #[test]
    fn annualized_vol_scales_with_sqrt_time() {
        let r = [0.01, -0.02, 0.015, 0.0, -0.005];
        let daily = annualized_volatility(&r, 1.0).unwrap();
        let annual = annualized_volatility(&r, 252.0).unwrap();
        assert!((annual - daily * 252.0_f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn bad_input_is_an_error_not_a_panic() {
        let empty: [f64; 0] = [];
        let nan = [0.01, f64::NAN];
        let inf = [0.01, f64::INFINITY];
        for r in [&empty[..], &nan[..], &inf[..]] {
            assert!(historical_var(r, 0.95).is_err());
            assert!(historical_cvar(r, 0.95).is_err());
            assert!(parametric_var(r, 0.95).is_err());
            assert!(sharpe_ratio(r, 0.0, 252.0).is_err());
            assert!(sortino_ratio(r, 0.0, 252.0).is_err());
            assert!(max_drawdown(r).is_err());
            assert!(annualized_volatility(r, 252.0).is_err());
        }
        let ok = [0.01, -0.02, 0.03];
        for c in [0.0, 1.0, -0.5, 1.5, f64::NAN] {
            assert!(historical_var(&ok, c).is_err());
            assert!(historical_cvar(&ok, c).is_err());
            assert!(parametric_var(&ok, c).is_err());
        }
        assert!(sharpe_ratio(&ok, 0.0, 0.0).is_err());
        assert!(sharpe_ratio(&ok, f64::NAN, 252.0).is_err());
        assert!(annualized_volatility(&[0.01], 252.0).is_err());
        assert!(parametric_var(&[0.01], 0.95).is_err());
    }

    #[test]
    fn scale_extremes_do_not_fake_zero_variance_or_overflow() {
        // mean 2e-200, sample std 1e-200 exactly: Sharpe = 2 * sqrt(ppy).
        let tiny = [1e-200, 2e-200, 3e-200];
        let s = sharpe_ratio(&tiny, 0.0, 4.0).unwrap();
        assert!((s - 4.0).abs() < 1e-12, "sharpe {s}");
        let vol = annualized_volatility(&tiny, 1.0).unwrap();
        assert!((vol / 1e-200 - 1.0).abs() < 1e-12);
        // Results that cannot be represented are errors, not inf/NaN.
        assert!(max_drawdown(&[1e308, 1e308, -0.5]).is_err());
        assert!(parametric_var(&[1e308, -1e308, 1e308], 0.95).is_err());
    }

    #[test]
    fn historical_tail_semantics() {
        // n = 7, c = 0.9: k = ceil(0.7) = 1 -> the single worst return.
        let r = [-0.05, -0.04, -0.03, 0.0, 0.01, 0.02, 0.03];
        assert_eq!(historical_var(&r, 0.9).unwrap(), 0.05);
        // n = 7, c = 0.5: k = ceil(3.5) = 4 -> fourth worst (0.0), tail rounds up.
        assert_eq!(historical_var(&r, 0.5).unwrap(), -0.0);
        assert!((historical_cvar(&r, 0.5).unwrap() - 0.03).abs() < 1e-15);
        // Ties.
        let t = [-0.02, -0.02, -0.02, 0.01];
        assert_eq!(historical_var(&t, 0.5).unwrap(), 0.02);
        assert_eq!(historical_cvar(&t, 0.5).unwrap(), 0.02);
    }

    #[test]
    fn single_observation_historical_works() {
        assert_eq!(historical_var(&[-0.03], 0.99).unwrap(), 0.03);
        assert_eq!(historical_cvar(&[-0.03], 0.99).unwrap(), 0.03);
    }
}
