//! # quantkit
//!
//! Options pricing, Greeks, implied volatility and portfolio risk metrics.
//!
//! Every fallible function returns [`Result`] with a [`QuantError`];
//! nothing panics on bad input.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod dist;
mod error;
mod option;

pub use error::{QuantError, Result};
pub use option::{Contract, ExerciseStyle, OptionBuilder, OptionParams, OptionType};
