//! Batch APIs: price, Greek and invert many options in one call.
//!
//! Results are returned in input order. Where a single item can fail
//! (implied volatility, binomial), the output is `Vec<Result<_>>`, so one bad
//! quote never discards the rest of the batch.
//!
//! Enable the `parallel` cargo feature to spread the work across threads
//! with [rayon](https://docs.rs/rayon). The function signatures and the
//! results are identical with and without it.
//!
//! [`grid`] builds the Cartesian product of input ranges, which is handy for
//! generating training data for pricing models.

use crate::bsm;
use crate::error::Result;
use crate::greeks::{self, Greeks};
use crate::implied_vol;
use crate::option::{Contract, ExerciseStyle, OptionParams, OptionType};

#[cfg(feature = "parallel")]
fn map<T: Sync, R: Send, F: Fn(&T) -> R + Sync + Send>(items: &[T], f: F) -> Vec<R> {
    use rayon::prelude::*;
    items.par_iter().map(f).collect()
}

#[cfg(not(feature = "parallel"))]
fn map<T: Sync, R: Send, F: Fn(&T) -> R + Sync + Send>(items: &[T], f: F) -> Vec<R> {
    items.iter().map(f).collect()
}

/// BSM prices for every option in `params`.
///
/// # Examples
///
/// ```
/// use quantkit::{batch, OptionParams, OptionType};
///
/// let p = OptionParams::builder().spot(100.0).strike(100.0).volatility(0.2).expiry(1.0).build()?;
/// let prices = batch::price(&[p, p], OptionType::Call);
/// assert_eq!(prices.len(), 2);
/// assert_eq!(prices[0], prices[1]);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn price(params: &[OptionParams], kind: OptionType) -> Vec<f64> {
    map(params, |p| bsm::price(p, kind))
}

/// Analytic BSM Greeks for every option in `params`.
///
/// # Examples
///
/// ```
/// use quantkit::{batch, OptionParams, OptionType};
///
/// let p = OptionParams::builder().spot(100.0).strike(100.0).volatility(0.2).expiry(1.0).build()?;
/// let g = batch::greeks(&[p], OptionType::Put);
/// assert!(g[0].delta < 0.0);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn greeks(params: &[OptionParams], kind: OptionType) -> Vec<Greeks> {
    map(params, |p| greeks::analytic(p, kind))
}

/// Binomial prices for every option in `params`. Each entry is its own
/// `Result`, because a given step count can be invalid for some inputs.
///
/// # Examples
///
/// ```
/// use quantkit::{batch, ExerciseStyle, OptionParams, OptionType};
///
/// let p = OptionParams::builder().spot(100.0).strike(100.0).volatility(0.2).expiry(1.0).build()?;
/// let out = batch::binomial_price(&[p], OptionType::Put, ExerciseStyle::American, 100);
/// assert!(out[0].is_ok());
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn binomial_price(
    params: &[OptionParams],
    kind: OptionType,
    style: ExerciseStyle,
    steps: usize,
) -> Vec<Result<f64>> {
    map(params, |p| crate::binomial::price(p, kind, style, steps))
}

/// One market quote to invert: a contract, its type and an observed price.
///
/// # Examples
///
/// ```
/// use quantkit::batch::IvQuote;
/// use quantkit::{OptionParams, OptionType};
///
/// let contract = OptionParams::builder().spot(21.0).strike(20.0).rate(0.1).expiry(0.25).build_contract()?;
/// let q = IvQuote { contract, kind: OptionType::Call, price: 1.875 };
/// assert_eq!(q.price, 1.875);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IvQuote {
    /// The contract terms (no volatility).
    pub contract: Contract,
    /// Call or put.
    pub kind: OptionType,
    /// Observed market price.
    pub price: f64,
}

/// Implied volatilities for many quotes, with default solver settings.
/// Failures (price out of bounds, no convergence, NaN) are reported per quote.
///
/// # Examples
///
/// ```
/// use quantkit::batch::{implied_vol, IvQuote};
/// use quantkit::{OptionParams, OptionType};
///
/// let contract = OptionParams::builder().spot(21.0).strike(20.0).rate(0.1).expiry(0.25).build_contract()?;
/// let good = IvQuote { contract, kind: OptionType::Call, price: 1.875 };
/// let bad = IvQuote { contract, kind: OptionType::Call, price: -1.0 };
/// let out = implied_vol(&[good, bad]);
/// assert!(out[0].is_ok() && out[1].is_err());
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn implied_vol(quotes: &[IvQuote]) -> Vec<Result<f64>> {
    map(quotes, |q| implied_vol::solve(&q.contract, q.kind, q.price))
}

/// Builds every combination of the given inputs as validated
/// [`OptionParams`].
///
/// Order is row-major with `spots` slowest and `vols` fastest: for each spot,
/// for each strike, for each expiry, for each vol.
///
/// # Errors
///
/// The first invalid combination (for example a non-positive strike in
/// `strikes`) is returned as an error.
///
/// # Examples
///
/// ```
/// use quantkit::batch::grid;
/// let g = grid(&[90.0, 100.0], &[100.0], &[0.5, 1.0], &[0.2, 0.3], 0.03, 0.0)?;
/// assert_eq!(g.len(), 2 * 1 * 2 * 2);
/// assert_eq!(g[0].spot(), 90.0);
/// assert_eq!(g[1].volatility(), 0.3); // vols vary fastest
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn grid(
    spots: &[f64],
    strikes: &[f64],
    expiries: &[f64],
    vols: &[f64],
    rate: f64,
    dividend_yield: f64,
) -> Result<Vec<OptionParams>> {
    let mut out = Vec::with_capacity(spots.len() * strikes.len() * expiries.len() * vols.len());
    for &s in spots {
        for &k in strikes {
            for &t in expiries {
                for &v in vols {
                    out.push(
                        OptionParams::builder()
                            .spot(s)
                            .strike(k)
                            .rate(rate)
                            .dividend_yield(dividend_yield)
                            .volatility(v)
                            .expiry(t)
                            .build()?,
                    );
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<OptionParams> {
        grid(
            &[80.0, 100.0, 120.0],
            &[90.0, 100.0, 110.0],
            &[0.25, 1.0],
            &[0.15, 0.4],
            0.03,
            0.01,
        )
        .unwrap()
    }

    #[test]
    fn batch_matches_scalar() {
        let ps = sample();
        let prices = price(&ps, OptionType::Call);
        let gs = greeks(&ps, OptionType::Put);
        let bin = binomial_price(&ps, OptionType::Put, ExerciseStyle::American, 50);
        assert_eq!(prices.len(), ps.len());
        for (i, p) in ps.iter().enumerate() {
            assert_eq!(prices[i], bsm::price(p, OptionType::Call));
            assert_eq!(gs[i], greeks::analytic(p, OptionType::Put));
            assert_eq!(
                *bin[i].as_ref().unwrap(),
                crate::binomial::price(p, OptionType::Put, ExerciseStyle::American, 50).unwrap()
            );
        }
    }

    #[test]
    fn iv_batch_isolates_failures() {
        let ps = sample();
        let mut quotes: Vec<IvQuote> = ps
            .iter()
            .map(|p| IvQuote {
                contract: *p.contract(),
                kind: OptionType::Call,
                price: bsm::price(p, OptionType::Call),
            })
            .collect();
        quotes[3].price = f64::NAN;
        quotes[5].price = -2.0;
        let out = implied_vol(&quotes);
        assert!(out[3].is_err() && out[5].is_err());
        for (i, r) in out.iter().enumerate() {
            if i != 3 && i != 5 {
                assert!(r.is_ok(), "quote {i}: {r:?}");
            }
        }
    }

    #[test]
    fn grid_order_size_and_errors() {
        let g = grid(&[1.0, 2.0], &[3.0, 4.0], &[5.0], &[0.1, 0.2], 0.0, 0.0).unwrap();
        assert_eq!(g.len(), 8);
        assert_eq!(
            (g[0].spot(), g[0].strike(), g[0].volatility()),
            (1.0, 3.0, 0.1)
        );
        assert_eq!((g[2].spot(), g[2].strike()), (1.0, 4.0));
        assert_eq!(g[4].spot(), 2.0);
        assert!(grid(&[1.0], &[-1.0], &[1.0], &[0.2], 0.0, 0.0).is_err());
        assert!(grid(&[], &[1.0], &[1.0], &[0.2], 0.0, 0.0)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn empty_batches() {
        assert!(price(&[], OptionType::Call).is_empty());
        assert!(implied_vol(&[]).is_empty());
    }
}
