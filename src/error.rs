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

    /// A market price lies outside the attainable range (the no-arbitrage
    /// bounds, narrowed to what the supported volatility range can produce),
    /// so no volatility can reproduce it.
    #[error("price {price} is outside the attainable range [{lower}, {upper}]")]
    PriceOutOfBounds {
        /// The supplied price.
        price: f64,
        /// Lowest price attainable (volatility -> 0).
        lower: f64,
        /// Highest price attainable (volatility -> infinity).
        upper: f64,
    },
}

/// A compact, stable status code for one row of a batch result.
///
/// The numeric values are part of the public contract (they are what the
/// Python bindings expose) and will not be renumbered.
///
/// | code | name |
/// |---|---|
/// | 0 | `Ok` |
/// | 1 | `InvalidInput` |
/// | 2 | `NonFinite` |
/// | 3 | `EmptyInput` |
/// | 4 | `NoConvergence` |
/// | 5 | `PriceOutOfBounds` |
/// | 255 | `Other` (a future error kind) |
///
/// # Examples
///
/// ```
/// use quantkit::{QuantError, StatusCode};
/// let e = QuantError::NonFinite { name: "spot" };
/// assert_eq!(e.status_code(), StatusCode::NonFinite);
/// assert_eq!(StatusCode::NonFinite as u8, 2);
/// assert_eq!(StatusCode::from_u8(4), Some(StatusCode::NoConvergence));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u8)]
pub enum StatusCode {
    /// The row succeeded.
    Ok = 0,
    /// An input was missing or outside its valid / supported range.
    InvalidInput = 1,
    /// An input or result was NaN or infinite.
    NonFinite = 2,
    /// A required slice was empty.
    EmptyInput = 3,
    /// A solver did not reach its tolerance.
    NoConvergence = 4,
    /// A price lies outside the attainable range.
    PriceOutOfBounds = 5,
    /// An error kind added after this version.
    Other = 255,
}

impl StatusCode {
    /// Short stable name, e.g. `"NoConvergence"`.
    pub fn name(self) -> &'static str {
        match self {
            StatusCode::Ok => "Ok",
            StatusCode::InvalidInput => "InvalidInput",
            StatusCode::NonFinite => "NonFinite",
            StatusCode::EmptyInput => "EmptyInput",
            StatusCode::NoConvergence => "NoConvergence",
            StatusCode::PriceOutOfBounds => "PriceOutOfBounds",
            StatusCode::Other => "Other",
        }
    }

    /// Converts a raw code back to a status, or `None` if unknown.
    pub fn from_u8(code: u8) -> Option<Self> {
        Some(match code {
            0 => StatusCode::Ok,
            1 => StatusCode::InvalidInput,
            2 => StatusCode::NonFinite,
            3 => StatusCode::EmptyInput,
            4 => StatusCode::NoConvergence,
            5 => StatusCode::PriceOutOfBounds,
            255 => StatusCode::Other,
            _ => return None,
        })
    }
}

impl QuantError {
    /// The [`StatusCode`] for this error.
    pub fn status_code(&self) -> StatusCode {
        match self {
            QuantError::InvalidInput { .. } => StatusCode::InvalidInput,
            QuantError::NonFinite { .. } => StatusCode::NonFinite,
            QuantError::EmptyInput { .. } => StatusCode::EmptyInput,
            QuantError::NoConvergence { .. } => StatusCode::NoConvergence,
            QuantError::PriceOutOfBounds { .. } => StatusCode::PriceOutOfBounds,
        }
    }
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
