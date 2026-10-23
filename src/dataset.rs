//! Reproducible synthetic option-pricing datasets for machine learning.
//!
//! [`generate`] draws option parameters from a seeded sampler (Latin
//! hypercube by default), prices every row in closed form with
//! Black-Scholes-Merton, attaches the five Greeks as labels, appends a fixed
//! set of boundary cases, and returns the rows together with the metadata
//! needed to reproduce and audit them: the full specification, the quantkit
//! version, conventions, a checksum of the data, and a report of financial
//! invariants checked on every row.
//!
//! # What this is and is not
//!
//! * Labels are **closed-form BSM prices for European options**. That is the
//!   usual ground truth for benchmarking neural pricers, but it is a
//!   synthetic world: no smile, no jumps, no bid-ask noise.
//! * The invariant report is a **self-consistency check** (put-call parity,
//!   no-arbitrage bounds, Greek signs, finiteness) computed from closed-form
//!   relations. It catches corrupt or out-of-domain rows; it is not an
//!   independent validation of the pricing formula, which is tested
//!   separately against reference values.
//! * Datasets are bitwise reproducible from `(spec, quantkit version)`: the
//!   sampler uses the crate's platform-independent [`mc::Rng`].
//!
//! # Conventions
//!
//! Sampling is log-uniform in moneyness `K/S`, expiry and volatility, and
//! linear in the rate and dividend yield. Vega is per 1.00 of volatility,
//! rho per 1.00 of rate, theta per year (see [`crate::greeks`]).

use crate::bsm;
use crate::error::{QuantError, Result};
use crate::greeks;
use crate::limits::*;
use crate::mc::Rng;
use crate::option::{OptionParams, OptionType};

/// Which option types to include.
///
/// # Examples
///
/// ```
/// use quantkit::dataset::KindMix;
/// assert_eq!(KindMix::default(), KindMix::Both);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KindMix {
    /// Calls only.
    CallsOnly,
    /// Puts only.
    PutsOnly,
    /// Roughly half calls, half puts.
    #[default]
    Both,
}

/// How parameter combinations are drawn.
///
/// # Examples
///
/// ```
/// use quantkit::dataset::Sampling;
/// assert_eq!(Sampling::default(), Sampling::LatinHypercube);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Sampling {
    /// Independent uniform draws in each dimension.
    Uniform,
    /// Each dimension is divided into `rows` equal strata and every stratum
    /// is used exactly once (better coverage than plain uniform sampling).
    #[default]
    LatinHypercube,
}

/// Dataset specification. Every field is part of the reproducibility key.
///
/// # Examples
///
/// ```
/// use quantkit::dataset::DatasetSpec;
/// let spec = DatasetSpec { rows: 100, seed: 7, ..DatasetSpec::default() };
/// assert_eq!(spec.rows, 100);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DatasetSpec {
    /// Number of sampled rows (boundary rows are added on top).
    pub rows: usize,
    /// Seed for the sampler.
    pub seed: u64,
    /// Reference spot price; strikes are `spot * moneyness`.
    pub spot: f64,
    /// Inclusive range of `K / S` (log-uniform).
    pub moneyness: (f64, f64),
    /// Inclusive range of years to expiry (log-uniform).
    pub expiry: (f64, f64),
    /// Inclusive range of volatility (log-uniform).
    pub volatility: (f64, f64),
    /// Inclusive range of the risk-free rate (linear).
    pub rate: (f64, f64),
    /// Inclusive range of the dividend yield (linear).
    pub dividend_yield: (f64, f64),
    /// Calls, puts, or both.
    pub kinds: KindMix,
    /// Sampling scheme.
    pub sampling: Sampling,
    /// Append the corners of the parameter box (see [`generate`]).
    pub include_boundary: bool,
}

impl Default for DatasetSpec {
    fn default() -> Self {
        Self {
            rows: 10_000,
            seed: 0,
            spot: 100.0,
            moneyness: (0.5, 2.0),
            expiry: (1.0 / 52.0, 3.0),
            volatility: (0.05, 1.0),
            rate: (-0.01, 0.06),
            dividend_yield: (0.0, 0.05),
            kinds: KindMix::Both,
            sampling: Sampling::LatinHypercube,
            include_boundary: true,
        }
    }
}

/// One labelled example.
///
/// # Examples
///
/// ```
/// use quantkit::dataset::{generate, DatasetSpec};
/// let ds = generate(&DatasetSpec { rows: 5, include_boundary: false, ..DatasetSpec::default() })?;
/// let row = &ds.rows[0];
/// assert!(row.price >= 0.0 && row.volatility > 0.0);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DatasetRow {
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
    /// Years to expiry.
    pub expiry: f64,
    /// `true` for a call, `false` for a put.
    pub is_call: bool,
    /// `true` for a deterministic boundary case, `false` for a sampled row.
    pub boundary: bool,
    /// BSM price.
    pub price: f64,
    /// BSM delta.
    pub delta: f64,
    /// BSM gamma.
    pub gamma: f64,
    /// BSM vega (per 1.00 volatility).
    pub vega: f64,
    /// BSM theta (per year).
    pub theta: f64,
    /// BSM rho (per 1.00 rate).
    pub rho: f64,
}

/// Counts from the financial-invariant checks run on every row.
///
/// # Examples
///
/// ```
/// use quantkit::dataset::{generate, DatasetSpec};
/// let ds = generate(&DatasetSpec { rows: 200, ..DatasetSpec::default() })?;
/// assert!(ds.metadata.invariants.all_passed());
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InvariantReport {
    /// Rows examined.
    pub rows_checked: usize,
    /// Rows with a non-finite price or Greek.
    pub non_finite: usize,
    /// Rows whose price is outside the no-arbitrage bounds.
    pub bound_violations: usize,
    /// Rows violating put-call parity (relative tolerance 1e-9).
    pub parity_violations: usize,
    /// Rows with delta outside its theoretical range.
    pub delta_violations: usize,
    /// Rows with negative gamma or vega.
    pub sign_violations: usize,
    /// Largest relative put-call parity error seen.
    pub max_parity_error: f64,
}

impl InvariantReport {
    /// `true` if no invariant was violated.
    pub fn all_passed(&self) -> bool {
        self.non_finite
            + self.bound_violations
            + self.parity_violations
            + self.delta_violations
            + self.sign_violations
            == 0
    }
}

/// Everything needed to reproduce and audit a dataset.
///
/// # Examples
///
/// ```
/// use quantkit::dataset::{generate, DatasetSpec};
/// let a = generate(&DatasetSpec { rows: 50, seed: 1, ..DatasetSpec::default() })?;
/// let b = generate(&DatasetSpec { rows: 50, seed: 1, ..DatasetSpec::default() })?;
/// assert_eq!(a.metadata.checksum, b.metadata.checksum);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DatasetMetadata {
    /// quantkit version that produced the data.
    pub quantkit_version: String,
    /// The specification used.
    pub spec: DatasetSpec,
    /// Description of the labelling model.
    pub label_model: String,
    /// Units of the Greek columns.
    pub conventions: String,
    /// Description of the random-number generator and sampler.
    pub sampler: String,
    /// Number of sampled rows.
    pub sampled_rows: usize,
    /// Number of boundary rows.
    pub boundary_rows: usize,
    /// FNV-1a 64-bit checksum over every row's bit patterns, for snapshot
    /// comparison.
    pub checksum: u64,
    /// Result of the invariant checks.
    pub invariants: InvariantReport,
}

/// A generated dataset.
///
/// # Examples
///
/// ```
/// use quantkit::dataset::{generate, DatasetSpec};
/// let ds = generate(&DatasetSpec { rows: 3, include_boundary: false, ..DatasetSpec::default() })?;
/// assert_eq!(ds.rows.len(), 3);
/// assert!(ds.to_csv().starts_with("spot,strike,"));
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Dataset {
    /// Provenance and audit information.
    pub metadata: DatasetMetadata,
    /// Boundary rows (if requested) followed by sampled rows.
    pub rows: Vec<DatasetRow>,
}

impl Dataset {
    /// Renders the rows as CSV with a header. Floats use Rust's shortest
    /// round-trip formatting, so parsing recovers the exact values.
    pub fn to_csv(&self) -> String {
        let mut out = String::from(
            "spot,strike,rate,dividend_yield,volatility,expiry,is_call,boundary,price,delta,gamma,vega,theta,rho\n",
        );
        for r in &self.rows {
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                r.spot,
                r.strike,
                r.rate,
                r.dividend_yield,
                r.volatility,
                r.expiry,
                u8::from(r.is_call),
                u8::from(r.boundary),
                r.price,
                r.delta,
                r.gamma,
                r.vega,
                r.theta,
                r.rho
            ));
        }
        out
    }
}

fn invalid(name: &'static str, reason: &'static str) -> QuantError {
    QuantError::InvalidInput { name, reason }
}

fn check_range(
    name: &'static str,
    (lo, hi): (f64, f64),
    min: f64,
    max: f64,
    positive: bool,
) -> Result<()> {
    if !lo.is_finite() || !hi.is_finite() {
        return Err(QuantError::NonFinite { name });
    }
    if lo > hi {
        return Err(invalid(name, "lower bound exceeds upper bound"));
    }
    if lo < min || hi > max || (positive && lo <= 0.0) {
        return Err(invalid(name, "range outside the supported domain (see quantkit::limits)"));
    }
    Ok(())
}

fn validate(spec: &DatasetSpec) -> Result<()> {
    if spec.rows == 0 || spec.rows > 50_000_000 {
        return Err(invalid("rows", "must be between 1 and 50 million"));
    }
    check_range("spot", (spec.spot, spec.spot), MIN_PRICE, MAX_PRICE, true)?;
    check_range("moneyness", spec.moneyness, 0.0, f64::MAX, true)?;
    let (k_lo, k_hi) = (
        spec.spot * spec.moneyness.0,
        spec.spot * spec.moneyness.1,
    );
    check_range("strike", (k_lo, k_hi), MIN_PRICE, MAX_PRICE, true)?;
    check_range("expiry", spec.expiry, MIN_EXPIRY, MAX_EXPIRY, true)?;
    check_range("volatility", spec.volatility, MIN_VOL, MAX_VOL, true)?;
    check_range("rate", spec.rate, -MAX_ABS_RATE, MAX_ABS_RATE, false)?;
    check_range(
        "dividend_yield",
        spec.dividend_yield,
        -MAX_ABS_RATE,
        MAX_ABS_RATE,
        false,
    )?;
    let worst_rate = spec.rate.0.abs().max(spec.rate.1.abs());
    let worst_div = spec.dividend_yield.0.abs().max(spec.dividend_yield.1.abs());
    if worst_rate.max(worst_div) * spec.expiry.1 > MAX_RATE_TIME {
        return Err(invalid(
            "rate",
            "|rate| * max expiry or |dividend_yield| * max expiry exceeds 500",
        ));
    }
    Ok(())
}

/// Latin-hypercube or uniform samples in `[0,1)`, one vector per dimension.
fn unit_samples(spec: &DatasetSpec, dims: usize) -> Vec<Vec<f64>> {
    let n = spec.rows;
    (0..dims)
        .map(|d| {
            let mut rng = Rng::for_stream(spec.seed, 0x4441_5441 + d as u64);
            match spec.sampling {
                Sampling::Uniform => (0..n).map(|_| rng.next_f64()).collect(),
                Sampling::LatinHypercube => {
                    let mut strata: Vec<usize> = (0..n).collect();
                    for i in (1..n).rev() {
                        // Unbiased index in 0..=i via 128-bit multiply-shift.
                        let j = ((u128::from(rng.next_u64()) * (i as u128 + 1)) >> 64) as usize;
                        strata.swap(i, j);
                    }
                    strata
                        .into_iter()
                        .map(|s| (s as f64 + rng.next_f64()) / n as f64)
                        .collect()
                }
            }
        })
        .collect()
}

fn log_lerp((lo, hi): (f64, f64), u: f64) -> f64 {
    (lo.ln() + u * (hi.ln() - lo.ln())).exp().clamp(lo, hi)
}

fn lin_lerp((lo, hi): (f64, f64), u: f64) -> f64 {
    (lo + u * (hi - lo)).clamp(lo, hi)
}

struct Draw {
    moneyness: f64,
    expiry: f64,
    volatility: f64,
    rate: f64,
    dividend_yield: f64,
    is_call: bool,
    boundary: bool,
}

fn label(spec: &DatasetSpec, d: &Draw) -> Result<DatasetRow> {
    let p = OptionParams::builder()
        .spot(spec.spot)
        .strike(spec.spot * d.moneyness)
        .rate(d.rate)
        .dividend_yield(d.dividend_yield)
        .volatility(d.volatility)
        .expiry(d.expiry)
        .build()?;
    let kind = if d.is_call {
        OptionType::Call
    } else {
        OptionType::Put
    };
    let g = greeks::analytic(&p, kind);
    Ok(DatasetRow {
        spot: p.spot(),
        strike: p.strike(),
        rate: p.rate(),
        dividend_yield: p.dividend_yield(),
        volatility: p.volatility(),
        expiry: p.expiry(),
        is_call: d.is_call,
        boundary: d.boundary,
        price: bsm::price(&p, kind),
        delta: g.delta,
        gamma: g.gamma,
        vega: g.vega,
        theta: g.theta,
        rho: g.rho,
    })
}

fn check_invariants(rows: &[DatasetRow]) -> InvariantReport {
    let mut rep = InvariantReport {
        rows_checked: rows.len(),
        ..InvariantReport::default()
    };
    for r in rows {
        let all = [r.price, r.delta, r.gamma, r.vega, r.theta, r.rho];
        if all.iter().any(|x| !x.is_finite()) {
            rep.non_finite += 1;
            continue;
        }
        let ds = r.spot * (-r.dividend_yield * r.expiry).exp();
        let dk = r.strike * (-r.rate * r.expiry).exp();
        let (lower, upper) = if r.is_call {
            ((ds - dk).max(0.0), ds)
        } else {
            ((dk - ds).max(0.0), dk)
        };
        let slack = 1e-9 * (ds + dk);
        if r.price < lower - slack || r.price > upper + slack {
            rep.bound_violations += 1;
        }
        // Parity: price the opposite option independently and compare.
        if let Ok(p) = OptionParams::builder()
            .spot(r.spot)
            .strike(r.strike)
            .rate(r.rate)
            .dividend_yield(r.dividend_yield)
            .volatility(r.volatility)
            .expiry(r.expiry)
            .build()
        {
            let (call, put) = if r.is_call {
                (r.price, bsm::price(&p, OptionType::Put))
            } else {
                (bsm::price(&p, OptionType::Call), r.price)
            };
            let err = ((call - put) - (ds - dk)).abs() / (ds + dk);
            rep.max_parity_error = rep.max_parity_error.max(err);
            if err > 1e-9 {
                rep.parity_violations += 1;
            }
        }
        let eq = (-r.dividend_yield * r.expiry).exp();
        let (d_lo, d_hi) = if r.is_call { (0.0, eq) } else { (-eq, 0.0) };
        if r.delta < d_lo - 1e-12 || r.delta > d_hi + 1e-12 {
            rep.delta_violations += 1;
        }
        if r.gamma < 0.0 || r.vega < 0.0 {
            rep.sign_violations += 1;
        }
    }
    rep
}

fn fnv1a(rows: &[DatasetRow]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for r in rows {
        for x in [
            r.spot,
            r.strike,
            r.rate,
            r.dividend_yield,
            r.volatility,
            r.expiry,
            r.price,
            r.delta,
            r.gamma,
            r.vega,
            r.theta,
            r.rho,
        ] {
            eat(&x.to_bits().to_le_bytes());
        }
        eat(&[u8::from(r.is_call), u8::from(r.boundary)]);
    }
    h
}

/// Generates a dataset.
///
/// Row order: boundary rows first (if requested), then sampled rows.
///
/// **Boundary rows** are the corners of the parameter box: moneyness in
/// `{lo, 1 (if inside the range), hi}`, expiry, volatility, rate and dividend
/// yield each at `{lo, hi}`, for every option type in [`KindMix`]. They make
/// sure the extremes of the supported region are always represented.
///
/// # Errors
///
/// [`QuantError::InvalidInput`] / [`QuantError::NonFinite`] if any range is
/// reversed, non-finite, or outside the supported domain ([`crate::limits`]),
/// if `rows` is 0 or above 50 million, or if `|rate| * expiry` could exceed
/// the supported bound.
///
/// # Examples
///
/// ```
/// use quantkit::dataset::{generate, DatasetSpec, Sampling};
/// let spec = DatasetSpec { rows: 1_000, seed: 42, sampling: Sampling::LatinHypercube, ..DatasetSpec::default() };
/// let ds = generate(&spec)?;
/// assert_eq!(ds.metadata.sampled_rows, 1_000);
/// assert!(ds.metadata.boundary_rows > 0);
/// assert!(ds.metadata.invariants.all_passed());
/// # Ok::<(), quantkit::QuantError>(())
/// ```
pub fn generate(spec: &DatasetSpec) -> Result<Dataset> {
    validate(spec)?;
    let kinds: &[bool] = match spec.kinds {
        KindMix::CallsOnly => &[true],
        KindMix::PutsOnly => &[false],
        KindMix::Both => &[true, false],
    };

    let mut draws: Vec<Draw> = Vec::new();
    if spec.include_boundary {
        let mut moneyness = vec![spec.moneyness.0];
        if spec.moneyness.0 < 1.0 && spec.moneyness.1 > 1.0 {
            moneyness.push(1.0);
        }
        if spec.moneyness.1 > spec.moneyness.0 {
            moneyness.push(spec.moneyness.1);
        }
        let ends = |(lo, hi): (f64, f64)| if lo < hi { vec![lo, hi] } else { vec![lo] };
        for &m in &moneyness {
            for &t in &ends(spec.expiry) {
                for &v in &ends(spec.volatility) {
                    for &r in &ends(spec.rate) {
                        for &q in &ends(spec.dividend_yield) {
                            for &is_call in kinds {
                                draws.push(Draw {
                                    moneyness: m,
                                    expiry: t,
                                    volatility: v,
                                    rate: r,
                                    dividend_yield: q,
                                    is_call,
                                    boundary: true,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    let boundary_rows = draws.len();

    let u = unit_samples(spec, 6);
    for i in 0..spec.rows {
        let is_call = match spec.kinds {
            KindMix::CallsOnly => true,
            KindMix::PutsOnly => false,
            KindMix::Both => u[5][i] < 0.5,
        };
        draws.push(Draw {
            moneyness: log_lerp(spec.moneyness, u[0][i]),
            expiry: log_lerp(spec.expiry, u[1][i]),
            volatility: log_lerp(spec.volatility, u[2][i]),
            rate: lin_lerp(spec.rate, u[3][i]),
            dividend_yield: lin_lerp(spec.dividend_yield, u[4][i]),
            is_call,
            boundary: false,
        });
    }

    let rows: Vec<DatasetRow> = draws
        .iter()
        .map(|d| label(spec, d))
        .collect::<Result<_>>()?;
    let invariants = check_invariants(&rows);
    let metadata = DatasetMetadata {
        quantkit_version: env!("CARGO_PKG_VERSION").to_string(),
        spec: *spec,
        label_model: "Black-Scholes-Merton, European, closed form, continuous dividend yield"
            .to_string(),
        conventions: "vega per 1.00 volatility; rho per 1.00 rate; theta per year; \
                      prices in units of the spot currency"
            .to_string(),
        sampler: "quantkit::mc::Rng (xoshiro256++, splitmix64-seeded, one stream per dimension); \
                  log-uniform moneyness/expiry/volatility, linear rate/dividend yield"
            .to_string(),
        sampled_rows: spec.rows,
        boundary_rows,
        checksum: fnv1a(&rows),
        invariants,
    };
    Ok(Dataset { metadata, rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small(seed: u64) -> DatasetSpec {
        DatasetSpec {
            rows: 400,
            seed,
            ..DatasetSpec::default()
        }
    }

    #[test]
    fn reproducible_and_seed_sensitive() {
        let a = generate(&small(1)).unwrap();
        let b = generate(&small(1)).unwrap();
        let c = generate(&small(2)).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.metadata.checksum, b.metadata.checksum);
        assert_ne!(a.metadata.checksum, c.metadata.checksum);
        assert_eq!(a.to_csv(), b.to_csv());
    }

    #[test]
    fn counts_and_invariants() {
        let ds = generate(&small(3)).unwrap();
        // 3 moneyness x 2 expiry x 2 vol x 2 rate x 2 div x 2 kinds = 96 corners.
        assert_eq!(ds.metadata.boundary_rows, 96);
        assert_eq!(ds.rows.len(), 96 + 400);
        assert!(ds.rows[..96].iter().all(|r| r.boundary));
        assert!(ds.rows[96..].iter().all(|r| !r.boundary));
        assert_eq!(ds.metadata.invariants.rows_checked, 496);
        assert!(ds.metadata.invariants.all_passed(), "{:?}", ds.metadata.invariants);
        assert!(ds.metadata.invariants.max_parity_error < 1e-9);
        let no_boundary = generate(&DatasetSpec {
            include_boundary: false,
            ..small(3)
        })
        .unwrap();
        assert_eq!(no_boundary.rows.len(), 400);
        assert_eq!(no_boundary.metadata.boundary_rows, 0);
    }

    #[test]
    fn ranges_are_respected_and_corners_are_present() {
        let spec = small(4);
        let ds = generate(&spec).unwrap();
        for r in &ds.rows {
            let m = r.strike / r.spot;
            assert!(m >= spec.moneyness.0 * (1.0 - 1e-12) && m <= spec.moneyness.1 * (1.0 + 1e-12));
            assert!(r.expiry >= spec.expiry.0 && r.expiry <= spec.expiry.1);
            assert!(r.volatility >= spec.volatility.0 && r.volatility <= spec.volatility.1);
            assert!(r.rate >= spec.rate.0 && r.rate <= spec.rate.1);
            assert!(r.dividend_yield >= spec.dividend_yield.0 && r.dividend_yield <= spec.dividend_yield.1);
        }
        let has = |f: &dyn Fn(&DatasetRow) -> bool| ds.rows.iter().any(|r| r.boundary && f(r));
        assert!(has(&|r| r.expiry == spec.expiry.0 && r.volatility == spec.volatility.1));
        assert!(has(&|r| r.strike == spec.spot && r.rate == spec.rate.0));
        assert!(has(&|r| r.is_call) && has(&|r| !r.is_call));
    }

    #[test]
    fn latin_hypercube_uses_every_stratum_exactly_once() {
        let spec = DatasetSpec {
            rows: 500,
            seed: 9,
            include_boundary: false,
            ..DatasetSpec::default()
        };
        let u = unit_samples(&spec, 6);
        for (d, col) in u.iter().enumerate() {
            let mut hits = vec![0u32; spec.rows];
            for &x in col {
                hits[(x * spec.rows as f64) as usize] += 1;
            }
            assert!(hits.iter().all(|&h| h == 1), "dimension {d} not stratified");
        }
        // Dimensions are not copies of each other.
        assert_ne!(u[0], u[1]);
    }

    #[test]
    fn kind_mix_is_honoured() {
        let calls = generate(&DatasetSpec {
            kinds: KindMix::CallsOnly,
            ..small(5)
        })
        .unwrap();
        assert!(calls.rows.iter().all(|r| r.is_call));
        let puts = generate(&DatasetSpec {
            kinds: KindMix::PutsOnly,
            sampling: Sampling::Uniform,
            ..small(5)
        })
        .unwrap();
        assert!(puts.rows.iter().all(|r| !r.is_call));
        let both = generate(&small(5)).unwrap();
        let n_calls = both.rows.iter().filter(|r| !r.boundary && r.is_call).count();
        assert_eq!(n_calls, 200); // Latin hypercube splits the kind dimension evenly
    }

    #[test]
    fn csv_round_trips_exactly() {
        let ds = generate(&DatasetSpec {
            rows: 20,
            ..small(6)
        })
        .unwrap();
        let csv = ds.to_csv();
        let mut lines = csv.lines();
        assert!(lines.next().unwrap().starts_with("spot,strike,rate"));
        for (line, row) in lines.zip(&ds.rows) {
            let f: Vec<&str> = line.split(',').collect();
            assert_eq!(f.len(), 14);
            assert_eq!(f[0].parse::<f64>().unwrap().to_bits(), row.spot.to_bits());
            assert_eq!(f[8].parse::<f64>().unwrap().to_bits(), row.price.to_bits());
            assert_eq!(f[13].parse::<f64>().unwrap().to_bits(), row.rho.to_bits());
            assert_eq!(f[6] == "1", row.is_call);
        }
    }

    #[test]
    fn rejects_bad_specs() {
        let ok = small(1);
        assert!(generate(&DatasetSpec { rows: 0, ..ok }).is_err());
        assert!(generate(&DatasetSpec { moneyness: (2.0, 0.5), ..ok }).is_err());
        assert!(generate(&DatasetSpec { moneyness: (0.0, 2.0), ..ok }).is_err());
        assert!(generate(&DatasetSpec { volatility: (f64::NAN, 1.0), ..ok }).is_err());
        assert!(generate(&DatasetSpec { volatility: (1e-12, 1.0), ..ok }).is_err());
        assert!(generate(&DatasetSpec { expiry: (0.1, 5e3), ..ok }).is_err());
        assert!(generate(&DatasetSpec { spot: -1.0, ..ok }).is_err());
        assert!(generate(&DatasetSpec { rate: (0.0, 5.0), expiry: (0.1, 200.0), ..ok }).is_err());
    }

    #[test]
    fn degenerate_ranges_are_allowed() {
        let spec = DatasetSpec {
            rows: 10,
            moneyness: (1.0, 1.0),
            expiry: (1.0, 1.0),
            volatility: (0.2, 0.2),
            rate: (0.03, 0.03),
            dividend_yield: (0.0, 0.0),
            ..DatasetSpec::default()
        };
        let ds = generate(&spec).unwrap();
        assert!(ds.rows.iter().all(|r| r.strike == 100.0 && r.volatility == 0.2));
        assert!(ds.metadata.invariants.all_passed());
    }
}
