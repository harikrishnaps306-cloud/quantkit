//! The supported input domain.
//!
//! [`OptionParams`](crate::OptionParams) rejects values outside these bounds.
//! Within them, closed-form prices and Greeks are finite and cannot overflow
//! or underflow to NaN. That is a deliberate trade: a library that accepts
//! `spot = 1e300` and returns NaN is harder to use safely than one that says
//! up front what it supports.
//!
//! Zero volatility and zero time to expiry are **not** accepted. Those are
//! deterministic limits (intrinsic value of the forward) where Black-Scholes
//! Greeks such as gamma and vega are mathematically undefined or
//! discontinuous, so quantkit refuses to manufacture finite numbers for them.
//! Compute intrinsic values directly if you need the limit.
//!
//! # Examples
//!
//! ```
//! use quantkit::{limits, OptionParams};
//! let too_small = OptionParams::builder()
//!     .spot(100.0).strike(100.0).volatility(limits::MIN_VOL / 10.0).expiry(1.0)
//!     .build();
//! assert!(too_small.is_err());
//! ```

/// Smallest accepted spot or strike.
pub const MIN_PRICE: f64 = 1e-8;
/// Largest accepted spot or strike.
pub const MAX_PRICE: f64 = 1e12;
/// Smallest accepted volatility (annualised, as a fraction).
pub const MIN_VOL: f64 = 1e-8;
/// Largest accepted volatility (1000 means 100,000%).
pub const MAX_VOL: f64 = 1e3;
/// Smallest accepted time to expiry in years.
pub const MIN_EXPIRY: f64 = 1e-8;
/// Largest accepted time to expiry in years.
pub const MAX_EXPIRY: f64 = 1e3;
/// Largest accepted absolute interest rate or dividend yield (10 means 1000%).
pub const MAX_ABS_RATE: f64 = 10.0;
/// Largest accepted `|rate| * expiry` and `|dividend_yield| * expiry`, which
/// keeps discount factors inside the range of `f64::exp`.
pub const MAX_RATE_TIME: f64 = 500.0;
