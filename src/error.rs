//! Error type shared by every fallible function in the crate.

use thiserror::Error;

/// Convenience alias for `Result<T, QuantError>`.
pub type Result<T> = std::result::Result<T, QuantError>;

/// Everything that can go wrong in `quantkit`.
///
/// Functions in this crate never panic on bad input; they return one of
/// these variants instead.
///
/// # Examples
///
/// ```
/// use quantkit::{OptionParams, QuantError};
///
/// let err = OptionParams::builder()
///     .spot(100.0).strike(100.0).volatility(-0.2).expiry(1.0)
///     .build()
///     .unwrap_err();
/// assert!(matches!(err, QuantError::InvalidInput { name: "volatility", .. }));
/// ```
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum QuantError {
    /// A parameter is finite but outside its valid range, or was not set.
    #[error("invalid input `{name}`: {reason}")]
    InvalidInput {
        /// Name of the offending parameter.
        name: &'static str,
        /// Why it was rejected.
        reason: &'static str,
    },

    /// A parameter or data point is NaN or infinite.
    #[error("input `{name}` is not finite (NaN or infinity)")]
    NonFinite {
        /// Name of the offending parameter.
        name: &'static str,
    },

    /// A required slice was empty.
    #[error("input `{name}` is empty")]
    EmptyInput {
        /// Name of the offending parameter.
        name: &'static str,
    },

    /// An iterative solver did not reach the requested tolerance.
    #[error("solver did not converge after {iterations} iterations (residual {residual:e})")]
    NoConvergence {
        /// Iterations performed.
        iterations: usize,
        /// Absolute price error at the best estimate.
        residual: f64,
    },

    /// A market price lies outside the no-arbitrage bounds, so no
    /// volatility can reproduce it.
    #[error("price {price} is outside the no-arbitrage range [{lower}, {upper}]")]
    PriceOutOfBounds {
        /// The supplied price.
        price: f64,
        /// Lowest price attainable (volatility -> 0).
        lower: f64,
        /// Highest price attainable (volatility -> infinity).
        upper: f64,
    },
}

/// Returns `Err(NonFinite)` unless `x` is finite.
pub(crate) fn check_finite(name: &'static str, x: f64) -> Result<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(QuantError::NonFinite { name })
    }
}

/// Returns `Err` unless `x` is finite and strictly positive.
pub(crate) fn check_positive(name: &'static str, x: f64) -> Result<f64> {
    check_finite(name, x)?;
    if x > 0.0 {
        Ok(x)
    } else {
        Err(QuantError::InvalidInput {
            name,
            reason: "must be strictly positive",
        })
    }
}

/// Returns `Err` unless `x` is finite and within `[lo, hi]`.
///
/// Non-positive values get the more specific "strictly positive" message when
/// `lo > 0`.
pub(crate) fn check_range(name: &'static str, x: f64, lo: f64, hi: f64) -> Result<f64> {
    check_finite(name, x)?;
    if lo > 0.0 && x <= 0.0 {
        return Err(QuantError::InvalidInput {
            name,
            reason: "must be strictly positive",
        });
    }
    if x < lo || x > hi {
        return Err(QuantError::InvalidInput {
            name,
            reason: "outside the supported range (see quantkit::limits)",
        });
    }
    Ok(x)
}
