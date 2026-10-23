//! # quantkit
//!
//! Options pricing, Greeks, implied volatility and portfolio risk metrics.
//!
//! * [`bsm`]: Black-Scholes-Merton European prices with a continuous dividend yield
//! * [`binomial`]: Cox-Ross-Rubinstein tree for American (and European) options
//! * [`greeks`]: analytic BSM Greeks and bump-and-reprice numerical Greeks
//! * [`implied_vol`]: Newton-Raphson solver with a bisection fallback
//! * [`risk`]: VaR, CVaR, Sharpe, Sortino, max drawdown, volatility
//! * [`dist`]: standard normal pdf, cdf and inverse cdf
//!
//! Inputs are validated once, when an [`OptionParams`] is built. Fallible
//! functions return [`Result`] with a [`QuantError`]; nothing panics on bad
//! input (negative volatility, zero expiry, NaN, empty return series).
//!
//! # Example
//!
//! ```
//! use quantkit::{bsm, greeks, OptionParams, OptionType};
//!
//! // Hull's textbook example.
//! let p = OptionParams::builder()
//!     .spot(42.0).strike(40.0).rate(0.10).volatility(0.20).expiry(0.5)
//!     .build()?;
//! let call = bsm::price(&p, OptionType::Call);
//! assert!((call - 4.76).abs() < 0.005);
//!
//! let g = greeks::analytic(&p, OptionType::Call);
//! assert!(g.delta > 0.0 && g.delta < 1.0);
//! # Ok::<(), quantkit::QuantError>(())
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod binomial;
pub mod bsm;
pub mod dist;
mod error;
pub mod greeks;
pub mod implied_vol;
mod option;
pub mod risk;

pub use error::{QuantError, Result};
pub use option::{Contract, ExerciseStyle, OptionBuilder, OptionParams, OptionType};
