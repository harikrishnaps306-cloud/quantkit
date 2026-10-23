//! Batch results with per-row diagnostics.
//!
//! The functions in [`crate::batch`] return plain numbers (or `Result`s). The
//! ones here take **raw, unvalidated rows** and return one result row per
//! input row, each carrying a [`StatusCode`], the original [`QuantError`] when
//! the row failed, and algorithm-specific diagnostics:
//!
//! | function | row type | diagnostics |
//! |---|---|---|
//! | [`price`] | [`PriceRow`] | status |
//! | [`greeks`] | [`GreeksRow`] | status |
//! | [`implied_vol`] | [`IvRow`] | solver residual, iteration counts, vega |
//! | [`mc_european`] | [`McRow`] | sampling standard error, paths used |
//!
//! A failed row never poisons the others. Numeric fields of a failed row are
//! `NaN` (iteration and residual fields keep whatever the failure reported,
//! for example the last residual of a solver that ran out of iterations), so
//! you always have the status to tell *why* a row is `NaN`.
//!
//! Row order matches input order. Results do not depend on the number of
//! threads. For [`mc_european`] each row gets its own seed derived from the
//! base seed and the row index, so a row's estimate depends on its position in
//! the batch but not on the other rows' contents.

use crate::batch::{map, map_min};
use crate::error::{QuantError, Result, StatusCode};
use crate::greeks::{self, Greeks};
use crate::implied_vol::{self, SolverConfig};
use crate::mc::{self, McConfig};
use crate::option::{Contract, OptionParams, OptionType};

/// One unvalidated option row. Validation (including the supported domain in
/// [`crate::limits`]) happens per row inside the batch functions.
///
/// # Examples
///
/// ```
/// use quantkit::detailed::RawOption;
/// let row = RawOption { spot: 100.0, strike: 100.0, rate: 0.03, dividend_yield: 0.0, volatility: 0.2, expiry: 1.0 };
/// assert!(row.validate().is_ok());
/// assert!(RawOption { volatility: -1.0, ..row }.validate().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RawOption {
    /// Spot price.
    pub spot: f64,
    /// Strike price.
    pub strike: f64,
    /// Risk-free rate.
    pub rate: f64,
    /// Continuous dividend yield.
    pub dividend_yield: f64,
    /// Volatility.
    pub volatility: f64,
    /// Time to expiry in years.
    pub expiry: f64,
}

impl RawOption {
    /// Validates into [`OptionParams`].
    ///
    /// # Errors
    ///
    /// Any [`QuantError`] from the builder.
    pub fn validate(&self) -> Result<OptionParams> {
        OptionParams::builder()
            .spot(self.spot)
            .strike(self.strike)
            .rate(self.rate)
            .dividend_yield(self.dividend_yield)
            .volatility(self.volatility)
            .expiry(self.expiry)
            .build()
    }
}

/// One unvalidated market quote to invert for implied volatility.
///
/// # Examples
///
/// ```
/// use quantkit::detailed::RawQuote;
/// let q = RawQuote { spot: 21.0, strike: 20.0, rate: 0.1, dividend_yield: 0.0, expiry: 0.25, price: 1.875 };
/// assert!(q.validate().is_ok());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RawQuote {
    /// Spot price.
    pub spot: f64,
    /// Strike price.
    pub strike: f64,
    /// Risk-free rate.
    pub rate: f64,
    /// Continuous dividend yield.
    pub dividend_yield: f64,
    /// Time to expiry in years.
    pub expiry: f64,
    /// Observed option price.
    pub price: f64,
}

impl RawQuote {
    /// Validates the contract terms (the price is checked by the solver).
    ///
    /// # Errors
    ///
    /// Any [`QuantError`] from the builder.
    pub fn validate(&self) -> Result<Contract> {
        OptionParams::builder()
            .spot(self.spot)
            .strike(self.strike)
            .rate(self.rate)
            .dividend_yield(self.dividend_yield)
            .expiry(self.expiry)
            .build_contract()
    }
}

/// Column-oriented batch result: `N` value columns, a status per row, and the
/// errors of failed rows (sparse).
///
/// This is the memory-lean form of the `*_detailed` row functions, intended for
/// array front ends such as the Python bindings. Failed rows hold `NaN` in
/// every value column and an entry in `errors`.
///
/// # Examples
///
/// ```
/// use quantkit::detailed::price_columns;
/// use quantkit::{OptionType, StatusCode};
/// let c = price_columns(&[100.0, 100.0], &[100.0, 100.0], &[0.0; 2], &[0.0; 2], &[0.2, -1.0], &[1.0; 2], OptionType::Call)?;
/// assert_eq!(c.status[0], StatusCode::Ok);
/// assert_eq!(c.status[1], StatusCode::InvalidInput);
/// assert!(c.values[0][1].is_nan());
/// assert_eq!(c.errors.len(), 1);
/// assert_eq!(c.errors[0].0, 1);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Columns<const N: usize> {
    /// Value columns, each of the input length.
    pub values: [Vec<f64>; N],
    /// Status per row.
    pub status: Vec<StatusCode>,
    /// `(row index, error)` for every failed row, in row order.
    pub errors: Vec<(usize, QuantError)>,
}

/// Rows per parallel work unit in the column functions.
const CHUNK: usize = 2048;

fn columns_map<const N: usize, F>(n: usize, f: F) -> Columns<N>
where
    F: Fn(usize) -> Result<[f64; N]> + Sync + Send,
{
    let chunks: Vec<(usize, usize)> = (0..n)
        .step_by(CHUNK)
        .map(|s| (s, (s + CHUNK).min(n)))
        .collect();
    type Part<const N: usize> = (Vec<[f64; N]>, Vec<StatusCode>, Vec<(usize, QuantError)>);
    let parts: Vec<Part<N>> = map_min(&chunks, 2, |&(start, end)| {
        let mut vals = Vec::with_capacity(end - start);
        let mut st = Vec::with_capacity(end - start);
        let mut errs = Vec::new();
        for i in start..end {
            match f(i) {
                Ok(v) => {
                    vals.push(v);
                    st.push(StatusCode::Ok);
                }
                Err(e) => {
                    vals.push([f64::NAN; N]);
                    st.push(e.status_code());
                    errs.push((i, e));
                }
            }
        }
        (vals, st, errs)
    });
    let mut out = Columns {
        values: std::array::from_fn(|_| Vec::with_capacity(n)),
        status: Vec::with_capacity(n),
        errors: Vec::new(),
    };
    for (vals, st, errs) in parts {
        for v in vals {
            for (col, x) in out.values.iter_mut().zip(v) {
                col.push(x);
            }
        }
        out.status.extend(st);
        out.errors.extend(errs);
    }
    out
}

fn same_len(lens: &[usize]) -> Result<usize> {
    let n = lens[0];
    if lens.iter().any(|&l| l != n) {
        return Err(QuantError::InvalidInput {
            name: "columns",
            reason: "all columns must have the same length",
        });
    }
    Ok(n)
}

fn row_params(cols: [&[f64]; 6], i: usize) -> Result<OptionParams> {
    RawOption {
        spot: cols[0][i],
        strike: cols[1][i],
        rate: cols[2][i],
        dividend_yield: cols[3][i],
        volatility: cols[4][i],
        expiry: cols[5][i],
    }
    .validate()
}

/// BSM prices from parallel columns, with per-row status. Equivalent to
/// [`price`] on the zipped rows, without building row structs.
///
/// # Errors
///
/// [`QuantError::InvalidInput`] if the columns differ in length.
pub fn price_columns(
    spot: &[f64],
    strike: &[f64],
    rate: &[f64],
    dividend_yield: &[f64],
    volatility: &[f64],
    expiry: &[f64],
    kind: OptionType,
) -> Result<Columns<1>> {
    let cols = [spot, strike, rate, dividend_yield, volatility, expiry];
    let n = same_len(&cols.map(<[f64]>::len))?;
    Ok(columns_map(n, |i| {
        row_params(cols, i).map(|p| [crate::bsm::price(&p, kind)])
    }))
}

/// Analytic Greeks from parallel columns, with per-row status. Value columns
/// are `[delta, gamma, vega, theta, rho]`.
///
/// # Errors
///
/// [`QuantError::InvalidInput`] if the columns differ in length.
///
/// # Examples
///
/// ```
/// use quantkit::detailed::greeks_columns;
/// use quantkit::OptionType;
/// let c = greeks_columns(&[100.0], &[100.0], &[0.0], &[0.0], &[0.2], &[1.0], OptionType::Call)?;
/// assert!(c.values[0][0] > 0.0 && c.values[1][0] > 0.0); // delta, gamma
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn greeks_columns(
    spot: &[f64],
    strike: &[f64],
    rate: &[f64],
    dividend_yield: &[f64],
    volatility: &[f64],
    expiry: &[f64],
    kind: OptionType,
) -> Result<Columns<5>> {
    let cols = [spot, strike, rate, dividend_yield, volatility, expiry];
    let n = same_len(&cols.map(<[f64]>::len))?;
    Ok(columns_map(n, |i| {
        row_params(cols, i).map(|p| {
            let g = greeks::analytic(&p, kind);
            [g.delta, g.gamma, g.vega, g.theta, g.rho]
        })
    }))
}

/// Result row for [`price`].
#[derive(Debug, Clone, PartialEq)]
pub struct PriceRow {
    /// `Ok` or the failure kind.
    pub status: StatusCode,
    /// The failure, when `status` is not `Ok`.
    pub error: Option<QuantError>,
    /// BSM price, or `NaN` on failure.
    pub price: f64,
}

/// BSM prices with per-row status.
///
/// # Examples
///
/// ```
/// use quantkit::detailed::{self, RawOption};
/// use quantkit::{OptionType, StatusCode};
/// let good = RawOption { spot: 100.0, strike: 100.0, rate: 0.0, dividend_yield: 0.0, volatility: 0.2, expiry: 1.0 };
/// let bad = RawOption { volatility: -0.2, ..good };
/// let rows = detailed::price(&[good, bad], OptionType::Call);
/// assert_eq!(rows[0].status, StatusCode::Ok);
/// assert_eq!(rows[1].status, StatusCode::InvalidInput);
/// assert!(rows[1].price.is_nan());
/// ```
pub fn price(rows: &[RawOption], kind: OptionType) -> Vec<PriceRow> {
    map(rows, |r| match r.validate() {
        Ok(p) => PriceRow {
            status: StatusCode::Ok,
            error: None,
            price: crate::bsm::price(&p, kind),
        },
        Err(e) => PriceRow {
            status: e.status_code(),
            error: Some(e),
            price: f64::NAN,
        },
    })
}

/// Result row for [`greeks`].
#[derive(Debug, Clone, PartialEq)]
pub struct GreeksRow {
    /// `Ok` or the failure kind.
    pub status: StatusCode,
    /// The failure, when `status` is not `Ok`.
    pub error: Option<QuantError>,
    /// Greeks (all `NaN` on failure).
    pub greeks: Greeks,
}

/// Analytic Greeks with per-row status.
///
/// # Examples
///
/// ```
/// use quantkit::detailed::{self, RawOption};
/// use quantkit::OptionType;
/// let r = RawOption { spot: 100.0, strike: 100.0, rate: 0.0, dividend_yield: 0.0, volatility: 0.2, expiry: 1.0 };
/// let rows = detailed::greeks(&[r], OptionType::Put);
/// assert!(rows[0].greeks.delta < 0.0);
/// ```
pub fn greeks(rows: &[RawOption], kind: OptionType) -> Vec<GreeksRow> {
    let nan = Greeks {
        delta: f64::NAN,
        gamma: f64::NAN,
        vega: f64::NAN,
        theta: f64::NAN,
        rho: f64::NAN,
    };
    map(rows, |r| match r.validate() {
        Ok(p) => GreeksRow {
            status: StatusCode::Ok,
            error: None,
            greeks: greeks::analytic(&p, kind),
        },
        Err(e) => GreeksRow {
            status: e.status_code(),
            error: Some(e),
            greeks: nan,
        },
    })
}

/// Result row for [`implied_vol`].
#[derive(Debug, Clone, PartialEq)]
pub struct IvRow {
    /// `Ok` or the failure kind.
    pub status: StatusCode,
    /// The failure, when `status` is not `Ok`.
    pub error: Option<QuantError>,
    /// Implied volatility, or `NaN` on failure.
    pub vol: f64,
    /// Absolute price residual at `vol`. On a `NoConvergence` failure this is
    /// the last residual reached; otherwise `NaN` on failure.
    pub residual: f64,
    /// Solver update steps (Newton plus bisection). On a `NoConvergence`
    /// failure, the steps performed before giving up; `0` for other failures.
    pub iterations: usize,
    /// How many steps were Newton steps (`0` on failure).
    pub newton_steps: usize,
    /// BSM vega at `vol`, a conditioning indicator (`NaN` on failure).
    pub vega: f64,
}

/// Implied volatilities with solver diagnostics for every row.
///
/// # Examples
///
/// ```
/// use quantkit::detailed::{self, RawQuote};
/// use quantkit::{implied_vol::SolverConfig, OptionType, StatusCode};
/// let ok = RawQuote { spot: 21.0, strike: 20.0, rate: 0.1, dividend_yield: 0.0, expiry: 0.25, price: 1.875 };
/// let impossible = RawQuote { price: 25.0, ..ok };
/// let rows = detailed::implied_vol(&[ok, impossible], OptionType::Call, SolverConfig::default());
/// assert_eq!(rows[0].status, StatusCode::Ok);
/// assert!(rows[0].residual < 1e-9 && rows[0].vega > 0.0);
/// assert_eq!(rows[1].status, StatusCode::PriceOutOfBounds);
/// ```
pub fn implied_vol(rows: &[RawQuote], kind: OptionType, config: SolverConfig) -> Vec<IvRow> {
    map(rows, |q| {
        let fail = |e: QuantError| {
            let (residual, iterations) = match &e {
                QuantError::NoConvergence {
                    residual,
                    iterations,
                } => (*residual, *iterations),
                _ => (f64::NAN, 0),
            };
            IvRow {
                status: e.status_code(),
                error: Some(e),
                vol: f64::NAN,
                residual,
                iterations,
                newton_steps: 0,
                vega: f64::NAN,
            }
        };
        let contract = match q.validate() {
            Ok(c) => c,
            Err(e) => return fail(e),
        };
        match implied_vol::solve_detailed(&contract, kind, q.price, config) {
            Ok(s) => IvRow {
                status: StatusCode::Ok,
                error: None,
                vol: s.vol,
                residual: s.residual,
                iterations: s.iterations,
                newton_steps: s.newton_steps,
                vega: s.vega,
            },
            Err(e) => fail(e),
        }
    })
}

/// Result row for [`mc_european`].
#[derive(Debug, Clone, PartialEq)]
pub struct McRow {
    /// `Ok` or the failure kind.
    pub status: StatusCode,
    /// The failure, when `status` is not `Ok`.
    pub error: Option<QuantError>,
    /// Monte Carlo price, or `NaN` on failure.
    pub price: f64,
    /// Sampling standard error (see [`crate::mc`] for its scope), or `NaN`.
    pub std_error: f64,
    /// The seed this row used.
    pub seed: u64,
}

/// Monte Carlo European prices with a standard error for every row.
///
/// Row `i` uses seed `mc::derive_seed(base.seed, i)`; `base.paths` and
/// `base.antithetic` apply to every row.
///
/// # Examples
///
/// ```
/// use quantkit::detailed::{self, RawOption};
/// use quantkit::{mc::McConfig, OptionType, StatusCode};
/// let r = RawOption { spot: 100.0, strike: 100.0, rate: 0.03, dividend_yield: 0.0, volatility: 0.2, expiry: 1.0 };
/// let cfg = McConfig { paths: 20_000, seed: 1, antithetic: true };
/// let rows = detailed::mc_european(&[r, r], OptionType::Call, cfg);
/// assert_eq!(rows[0].status, StatusCode::Ok);
/// assert!(rows[0].std_error > 0.0);
/// assert_ne!(rows[0].seed, rows[1].seed); // independent estimates per row
/// ```
pub fn mc_european(rows: &[RawOption], kind: OptionType, base: McConfig) -> Vec<McRow> {
    let indexed: Vec<(usize, RawOption)> = rows.iter().copied().enumerate().collect();
    map(&indexed, |(i, r)| {
        let seed = mc::derive_seed(base.seed, *i as u64);
        let outcome = r
            .validate()
            .and_then(|p| mc::european(&p, kind, McConfig { seed, ..base }));
        match outcome {
            Ok(m) => McRow {
                status: StatusCode::Ok,
                error: None,
                price: m.price,
                std_error: m.std_error,
                seed,
            },
            Err(e) => McRow {
                status: e.status_code(),
                error: Some(e),
                price: f64::NAN,
                std_error: f64::NAN,
                seed,
            },
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bsm, OptionType};

    fn good() -> RawOption {
        RawOption {
            spot: 100.0,
            strike: 105.0,
            rate: 0.03,
            dividend_yield: 0.01,
            volatility: 0.25,
            expiry: 0.75,
        }
    }

    #[test]
    fn rows_match_scalar_functions_and_isolate_failures() {
        let mut rows = vec![good(); 6];
        rows[2].volatility = -1.0;
        rows[4].spot = f64::NAN;
        let p = price(&rows, OptionType::Call);
        let g = greeks(&rows, OptionType::Call);
        let exact = bsm::price(&good().validate().unwrap(), OptionType::Call);
        for (i, (pr, gr)) in p.iter().zip(&g).enumerate() {
            if i == 2 || i == 4 {
                assert_ne!(pr.status, StatusCode::Ok);
                assert!(pr.price.is_nan() && gr.greeks.delta.is_nan());
                assert!(pr.error.is_some());
            } else {
                assert_eq!(pr.status, StatusCode::Ok);
                assert_eq!(pr.price, exact);
                assert!(pr.error.is_none() && gr.greeks.gamma > 0.0);
            }
        }
        assert_eq!(p[2].status, StatusCode::InvalidInput);
        assert_eq!(p[4].status, StatusCode::NonFinite);
    }

    #[test]
    fn iv_rows_report_distinct_failure_kinds_and_diagnostics() {
        let c = good();
        let p = c.validate().unwrap();
        let ok_price = bsm::price(&p, OptionType::Call);
        let base = RawQuote {
            spot: c.spot,
            strike: c.strike,
            rate: c.rate,
            dividend_yield: c.dividend_yield,
            expiry: c.expiry,
            price: ok_price,
        };
        let rows_in = [
            base,
            RawQuote {
                price: -1.0,
                ..base
            },
            RawQuote {
                price: f64::NAN,
                ..base
            },
            RawQuote {
                expiry: 0.0,
                ..base
            },
            RawQuote {
                price: 1e-300,
                strike: 400.0,
                ..base
            },
        ];
        let rows = implied_vol(&rows_in, OptionType::Call, SolverConfig::default());
        assert_eq!(rows[0].status, StatusCode::Ok);
        assert!((rows[0].vol - 0.25).abs() < 1e-8);
        assert!(rows[0].residual <= 1e-12 * ok_price * 8.0 + 1e-12);
        assert!(rows[0].newton_steps <= rows[0].iterations);
        assert_eq!(rows[1].status, StatusCode::PriceOutOfBounds);
        assert_eq!(rows[2].status, StatusCode::NonFinite);
        assert_eq!(rows[3].status, StatusCode::InvalidInput);
        // A price the model cannot resolve fails visibly instead of returning garbage.
        assert_ne!(rows[4].status, StatusCode::Ok);
        assert!(rows[4].vol.is_nan());
    }

    #[test]
    fn mc_rows_are_reproducible_and_independent_of_neighbours() {
        let cfg = McConfig {
            paths: 8_192,
            seed: 11,
            antithetic: true,
        };
        let a = [good(), good(), good()];
        let mut b = a;
        b[2].volatility = 0.4; // change a *later* row
        let ra = mc_european(&a, OptionType::Call, cfg);
        let rb = mc_european(&b, OptionType::Call, cfg);
        assert_eq!(ra[0], rb[0]);
        assert_eq!(ra[1], rb[1]);
        assert_ne!(ra[2].price, rb[2].price);
        assert_ne!(ra[0].seed, ra[1].seed);
        assert_ne!(ra[0].price, ra[1].price);
        assert!(ra.iter().all(|r| r.std_error > 0.0));
    }

    #[test]
    fn column_functions_match_row_functions() {
        let mut rows = vec![good(); 5000];
        for (i, r) in rows.iter_mut().enumerate() {
            r.spot = 80.0 + (i % 50) as f64;
            r.volatility = 0.1 + (i % 30) as f64 * 0.01;
        }
        rows[7].volatility = -1.0;
        rows[4999].expiry = f64::NAN;
        let col = |f: fn(&RawOption) -> f64| rows.iter().map(f).collect::<Vec<_>>();
        let cols = (
            col(|r| r.spot),
            col(|r| r.strike),
            col(|r| r.rate),
            col(|r| r.dividend_yield),
            col(|r| r.volatility),
            col(|r| r.expiry),
        );
        let c = price_columns(
            &cols.0,
            &cols.1,
            &cols.2,
            &cols.3,
            &cols.4,
            &cols.5,
            OptionType::Put,
        )
        .unwrap();
        let g = greeks_columns(
            &cols.0,
            &cols.1,
            &cols.2,
            &cols.3,
            &cols.4,
            &cols.5,
            OptionType::Put,
        )
        .unwrap();
        let p_rows = price(&rows, OptionType::Put);
        let g_rows = greeks(&rows, OptionType::Put);
        for i in 0..rows.len() {
            assert_eq!(c.status[i], p_rows[i].status);
            assert_eq!(c.values[0][i].to_bits(), p_rows[i].price.to_bits());
            assert_eq!(g.status[i], g_rows[i].status);
            assert_eq!(g.values[0][i].to_bits(), g_rows[i].greeks.delta.to_bits());
            assert_eq!(g.values[4][i].to_bits(), g_rows[i].greeks.rho.to_bits());
        }
        let failed: Vec<usize> = c.errors.iter().map(|(i, _)| *i).collect();
        assert_eq!(failed, vec![7, 4999]);
        assert!(price_columns(
            &[1.0],
            &[1.0, 2.0],
            &[0.0],
            &[0.0],
            &[0.2],
            &[1.0],
            OptionType::Call
        )
        .is_err());
    }

    #[test]
    fn empty_input_gives_empty_output() {
        assert!(price(&[], OptionType::Call).is_empty());
        assert!(implied_vol(&[], OptionType::Call, SolverConfig::default()).is_empty());
    }
}
