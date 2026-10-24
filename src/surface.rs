//! Implied-volatility surfaces: construction from quotes, interpolation,
//! static-arbitrage checks and SVI smile fitting.
//!
//! # Interpolation convention
//!
//! A [`VolSurface`] stores one smile per expiry, indexed by **log-moneyness**
//! `k = ln(K / F(T))` with forward `F(T) = S e^{(r - q) T}`.
//!
//! * Across strikes, total variance `w = sigma^2 T` is interpolated linearly
//!   in `k`; outside the quoted strikes the edge volatility is held flat.
//! * Across expiries, `w` is interpolated linearly in `T` at fixed `k`
//!   (at each fixed `k` this keeps total variance non-decreasing in `T`
//!   between two slices whose interpolated values at that `k` are ordered;
//!   it does not make the surface arbitrage-free);
//!   before the first and after the last expiry the volatility is held flat.
//!
//! These are simple, predictable rules rather than a calibrated model. Use
//! [`VolSurface::check_arbitrage`] to find quotes that admit static
//! arbitrage, and [`fit_svi`] to get a smooth parametric smile.
//!
//! # References
//!
//! * J. Gatheral, "A parsimonious arbitrage-free implied volatility
//!   parameterization with application to the valuation of volatility
//!   derivatives" (2004), for the raw SVI form.
//! * J. Gatheral and A. Jacquier, "Arbitrage-free SVI volatility surfaces"
//!   (2014), for the no-butterfly-arbitrage density `g(k)` and the
//!   calendar-spread condition on total variance.
//! * Zeliade Systems, "Quasi-explicit calibration of Gatheral's SVI model"
//!   (2009), for the fitting approach used here.

use crate::bsm;
use crate::error::{check_finite, check_positive, QuantError, Result};
use crate::implied_vol;
use crate::linalg::{nelder_mead, solve3};
use crate::option::{OptionParams, OptionType};

/// One smile: volatilities at increasing strikes for a single expiry.
///
/// # Examples
///
/// ```
/// use quantkit::surface::SurfaceSlice;
/// let s = SurfaceSlice { expiry: 0.5, strikes: vec![90.0, 100.0, 110.0], vols: vec![0.25, 0.20, 0.22] };
/// assert_eq!(s.strikes.len(), s.vols.len());
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SurfaceSlice {
    /// Time to expiry in years (> 0).
    pub expiry: f64,
    /// Strictly increasing strikes (> 0).
    pub strikes: Vec<f64>,
    /// Implied volatility at each strike (> 0).
    pub vols: Vec<f64>,
}

/// A market option price to be inverted into a surface point.
///
/// # Examples
///
/// ```
/// use quantkit::surface::PriceQuote;
/// use quantkit::OptionType;
/// let q = PriceQuote { strike: 100.0, expiry: 0.5, kind: OptionType::Call, price: 7.0 };
/// assert_eq!(q.strike, 100.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PriceQuote {
    /// Strike.
    pub strike: f64,
    /// Time to expiry in years.
    pub expiry: f64,
    /// Call or put. Out-of-the-money quotes are the best conditioned.
    pub kind: OptionType,
    /// Observed price.
    pub price: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct Smile {
    expiry: f64,
    strikes: Vec<f64>,
    /// Log-moneyness `ln(K / F(T))` at each strike.
    ks: Vec<f64>,
    vols: Vec<f64>,
}

impl Smile {
    /// Volatility at log-moneyness `k`: linear in total variance, flat outside.
    fn vol_at(&self, k: f64) -> f64 {
        let n = self.ks.len();
        if k <= self.ks[0] {
            return self.vols[0];
        }
        if k >= self.ks[n - 1] {
            return self.vols[n - 1];
        }
        let hi = self.ks.partition_point(|&x| x <= k);
        let lo = hi - 1;
        let f = (k - self.ks[lo]) / (self.ks[hi] - self.ks[lo]);
        let w = (1.0 - f) * self.vols[lo].powi(2) + f * self.vols[hi].powi(2);
        w.sqrt()
    }
}

/// A calendar-spread arbitrage: total variance decreases with expiry.
///
/// # Examples
///
/// ```
/// use quantkit::surface::CalendarViolation;
/// fn shortfall(v: &CalendarViolation) -> f64 { v.total_variance_short - v.total_variance_long }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalendarViolation {
    /// Log-moneyness where it occurs.
    pub log_moneyness: f64,
    /// Earlier expiry.
    pub expiry_short: f64,
    /// Later expiry.
    pub expiry_long: f64,
    /// Total variance at the earlier expiry.
    pub total_variance_short: f64,
    /// Total variance at the later expiry (smaller than the earlier one).
    pub total_variance_long: f64,
}

/// A butterfly or vertical-spread arbitrage in one smile: call prices implied
/// by the quoted vols are not decreasing and convex in strike.
///
/// # Examples
///
/// ```
/// use quantkit::surface::ButterflyViolation;
/// fn size(v: &ButterflyViolation) -> f64 { v.excess }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ButterflyViolation {
    /// Expiry of the smile.
    pub expiry: f64,
    /// Strike at which the condition fails.
    pub strike: f64,
    /// By how much the condition is violated, in price units.
    pub excess: f64,
}

/// Static-arbitrage findings for a surface.
///
/// # Examples
///
/// ```
/// use quantkit::surface::ArbitrageReport;
/// let r = ArbitrageReport::default();
/// assert!(r.is_clean());
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ArbitrageReport {
    /// What was examined. A clean report only speaks for this scope.
    pub scope: ArbitrageScope,
    /// Calendar-spread violations.
    pub calendar: Vec<CalendarViolation>,
    /// Butterfly / vertical-spread violations.
    pub butterfly: Vec<ButterflyViolation>,
    /// Number of grid or quote evaluations that could not produce finite prices.
    pub evaluation_errors: usize,
}

/// Which points an [`ArbitrageReport`] covers. Neither scope is a proof of a
/// globally arbitrage-free surface.
///
/// # Examples
///
/// ```
/// use quantkit::surface::ArbitrageScope;
/// assert_eq!(ArbitrageScope::default(), ArbitrageScope::QuotedNodes);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ArbitrageScope {
    /// Only the quoted (strike, expiry) nodes.
    #[default]
    QuotedNodes,
    /// A finite grid of points of the *interpolated* surface.
    SampledGrid {
        /// Strikes sampled per expiry (uniform in log-moneyness).
        strikes_per_expiry: usize,
        /// Number of expiries sampled (quoted expiries plus interior points).
        expiries: usize,
    },
}

impl ArbitrageReport {
    /// `true` when no violations were found **within [`ArbitrageReport::scope`]**.
    /// This does not certify the surface is arbitrage-free elsewhere.
    pub fn is_clean(&self) -> bool {
        self.calendar.is_empty() && self.butterfly.is_empty() && self.evaluation_errors == 0
    }
}

/// An implied-volatility surface. See the [module docs](self) for the
/// interpolation rules.
///
/// # Examples
///
/// ```
/// use quantkit::surface::{SurfaceSlice, VolSurface};
///
/// let surface = VolSurface::new(100.0, 0.02, 0.0, vec![
///     SurfaceSlice { expiry: 0.25, strikes: vec![90.0, 100.0, 110.0], vols: vec![0.26, 0.21, 0.23] },
///     SurfaceSlice { expiry: 1.00, strikes: vec![80.0, 100.0, 120.0], vols: vec![0.27, 0.22, 0.24] },
/// ])?;
/// let v = surface.vol(100.0, 0.5)?;
/// assert!(v > 0.19 && v < 0.25);
/// assert!(surface.check_arbitrage().calendar.is_empty());
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct VolSurface {
    spot: f64,
    rate: f64,
    dividend_yield: f64,
    smiles: Vec<Smile>,
}

impl VolSurface {
    /// Builds a surface from smiles.
    ///
    /// # Errors
    ///
    /// * [`QuantError::EmptyInput`] if there are no slices or a slice has no strikes.
    /// * [`QuantError::InvalidInput`] / [`QuantError::NonFinite`] for non-positive
    ///   or non-finite spot, expiries, strikes or vols, for expiries or strikes that
    ///   are not strictly increasing, or for mismatched `strikes` / `vols` lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::surface::{SurfaceSlice, VolSurface};
    /// let bad = SurfaceSlice { expiry: 1.0, strikes: vec![100.0, 90.0], vols: vec![0.2, 0.2] };
    /// assert!(VolSurface::new(100.0, 0.0, 0.0, vec![bad]).is_err());
    /// ```
    pub fn new(
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        slices: Vec<SurfaceSlice>,
    ) -> Result<Self> {
        check_positive("spot", spot)?;
        check_finite("rate", rate)?;
        check_finite("dividend_yield", dividend_yield)?;
        if slices.is_empty() {
            return Err(QuantError::EmptyInput { name: "slices" });
        }
        let mut smiles: Vec<Smile> = Vec::with_capacity(slices.len());
        for sl in slices {
            check_positive("expiry", sl.expiry)?;
            if let Some(prev) = smiles.last() {
                if sl.expiry <= prev.expiry {
                    return Err(QuantError::InvalidInput {
                        name: "expiry",
                        reason: "slice expiries must be strictly increasing",
                    });
                }
            }
            if sl.strikes.is_empty() {
                return Err(QuantError::EmptyInput { name: "strikes" });
            }
            if sl.strikes.len() != sl.vols.len() {
                return Err(QuantError::InvalidInput {
                    name: "vols",
                    reason: "must have the same length as strikes",
                });
            }
            for &k in &sl.strikes {
                check_positive("strike", k)?;
            }
            for &v in &sl.vols {
                check_positive("vol", v)?;
            }
            if sl.strikes.windows(2).any(|w| w[1] <= w[0]) {
                return Err(QuantError::InvalidInput {
                    name: "strikes",
                    reason: "must be strictly increasing",
                });
            }
            // Reuse the option-domain validation (limits, rate * expiry bound)
            // for every node, so later pricing can never fail.
            for (&k, &v) in sl.strikes.iter().zip(&sl.vols) {
                OptionParams::builder()
                    .spot(spot)
                    .strike(k)
                    .rate(rate)
                    .dividend_yield(dividend_yield)
                    .volatility(v)
                    .expiry(sl.expiry)
                    .build()?;
            }
            let ks: Vec<f64> = sl
                .strikes
                .iter()
                .map(|&k| k.ln() - spot.ln() - (rate - dividend_yield) * sl.expiry)
                .collect();
            if ks.windows(2).any(|w| w[1] <= w[0]) {
                return Err(QuantError::InvalidInput { name: "strikes", reason: "log-moneyness coordinates are indistinguishable at floating-point precision" });
            }
            smiles.push(Smile {
                expiry: sl.expiry,
                strikes: sl.strikes,
                ks,
                vols: sl.vols,
            });
        }
        Ok(Self {
            spot,
            rate,
            dividend_yield,
            smiles,
        })
    }

    /// Builds a surface by inverting option prices with the BSM implied-vol
    /// solver. Quotes are grouped by identical expiry.
    ///
    /// # Errors
    ///
    /// Any quote that cannot be inverted (price out of bounds, no convergence)
    /// fails the build with that quote's error; plus the errors of
    /// [`VolSurface::new`], for example duplicated strikes within an expiry.
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::surface::{PriceQuote, VolSurface};
    /// use quantkit::{bsm, OptionParams, OptionType};
    ///
    /// let mut quotes = Vec::new();
    /// for strike in [90.0, 100.0, 110.0] {
    ///     let p = OptionParams::builder().spot(100.0).strike(strike).rate(0.02).volatility(0.2).expiry(0.5).build()?;
    ///     quotes.push(PriceQuote { strike, expiry: 0.5, kind: OptionType::Call, price: bsm::price(&p, OptionType::Call) });
    /// }
    /// let surface = VolSurface::from_quotes(100.0, 0.02, 0.0, &quotes)?;
    /// assert!((surface.vol(100.0, 0.5)? - 0.2).abs() < 1e-8);
    /// # Ok::<(), quantkit::QuantError>(())
    /// ```
    pub fn from_quotes(
        spot: f64,
        rate: f64,
        dividend_yield: f64,
        quotes: &[PriceQuote],
    ) -> Result<Self> {
        let mut sorted: Vec<PriceQuote> = quotes.to_vec();
        for q in &sorted {
            check_finite("expiry", q.expiry)?;
            check_finite("strike", q.strike)?;
        }
        sorted.sort_by(|a, b| {
            a.expiry
                .total_cmp(&b.expiry)
                .then(a.strike.total_cmp(&b.strike))
        });
        let mut slices: Vec<SurfaceSlice> = Vec::new();
        for q in sorted {
            let contract = OptionParams::builder()
                .spot(spot)
                .strike(q.strike)
                .rate(rate)
                .dividend_yield(dividend_yield)
                .expiry(q.expiry)
                .build_contract()?;
            let iv = implied_vol::solve(&contract, q.kind, q.price)?;
            match slices.last_mut() {
                Some(s) if s.expiry == q.expiry => {
                    s.strikes.push(q.strike);
                    s.vols.push(iv);
                }
                _ => slices.push(SurfaceSlice {
                    expiry: q.expiry,
                    strikes: vec![q.strike],
                    vols: vec![iv],
                }),
            }
        }
        Self::new(spot, rate, dividend_yield, slices)
    }

    fn log_forward(&self, t: f64) -> f64 {
        self.spot.ln() + (self.rate - self.dividend_yield) * t
    }

    /// Interpolated implied volatility at `(strike, expiry)`.
    ///
    /// # Errors
    ///
    /// [`QuantError::InvalidInput`] or [`QuantError::NonFinite`] unless
    /// `strike` and `expiry` are finite and positive.
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::surface::{SurfaceSlice, VolSurface};
    /// let s = VolSurface::new(100.0, 0.0, 0.0, vec![
    ///     SurfaceSlice { expiry: 1.0, strikes: vec![100.0], vols: vec![0.2] },
    /// ])?;
    /// assert_eq!(s.vol(80.0, 3.0)?, 0.2); // flat extrapolation
    /// assert!(s.vol(-1.0, 1.0).is_err());
    /// # Ok::<(), quantkit::QuantError>(())
    /// ```
    pub fn vol(&self, strike: f64, expiry: f64) -> Result<f64> {
        check_positive("strike", strike)?;
        check_positive("expiry", expiry)?;
        let k = strike.ln() - self.spot.ln() - (self.rate - self.dividend_yield) * expiry;
        check_finite("log_moneyness", k)?;
        let first = &self.smiles[0];
        let last = &self.smiles[self.smiles.len() - 1];
        if expiry <= first.expiry {
            return Ok(first.vol_at(k));
        }
        if expiry >= last.expiry {
            return Ok(last.vol_at(k));
        }
        let hi = self.smiles.partition_point(|s| s.expiry <= expiry);
        let (a, b) = (&self.smiles[hi - 1], &self.smiles[hi]);
        let (wa, wb) = (
            a.vol_at(k).powi(2) * a.expiry,
            b.vol_at(k).powi(2) * b.expiry,
        );
        let f = (expiry - a.expiry) / (b.expiry - a.expiry);
        Ok(((wa + f * (wb - wa)) / expiry).sqrt())
    }

    /// Looks for static arbitrage in the quoted nodes.
    ///
    /// * **Calendar:** at every shared log-moneyness, total variance must not
    ///   fall as expiry increases (checked over the strike range quoted for
    ///   both expiries).
    /// * **Butterfly / vertical:** within each smile, the BSM call prices
    ///   implied by the quoted vols must be non-increasing and convex in
    ///   strike.
    ///
    /// * **Vertical spread:** the call-price slope in strike is bounded by
    ///   the discount factor, `C(K1) - C(K2) <= e^{-rT} (K2 - K1)`.
    ///
    /// Forwards use `F(T) = S e^{(r - q) T}` and discounting uses `e^{-rT}`,
    /// with the surface's own `rate` and `dividend_yield`. Calendar checks
    /// compare total variance `sigma^2 T` at equal *forward* log-moneyness
    /// `ln(K / F(T))`, which is the right comparison under deterministic
    /// rates and dividends.
    ///
    /// **Scope: quoted nodes only** (`ArbitrageScope::QuotedNodes`). A clean
    /// report does not prove the interpolated surface is arbitrage-free
    /// between nodes (see [`VolSurface::check_arbitrage_sampled`]) and is
    /// never a global guarantee.
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::surface::{SurfaceSlice, VolSurface};
    /// // Longer expiry with *lower* total variance: calendar arbitrage.
    /// let s = VolSurface::new(100.0, 0.0, 0.0, vec![
    ///     SurfaceSlice { expiry: 0.5, strikes: vec![100.0], vols: vec![0.40] },
    ///     SurfaceSlice { expiry: 1.0, strikes: vec![100.0], vols: vec![0.20] },
    /// ])?;
    /// assert!(!s.check_arbitrage().calendar.is_empty());
    /// # Ok::<(), quantkit::QuantError>(())
    /// ```
    pub fn check_arbitrage(&self) -> ArbitrageReport {
        let mut report = ArbitrageReport::default();

        for pair in self.smiles.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            let lo = a.ks[0].max(b.ks[0]);
            let hi = a.ks[a.ks.len() - 1].min(b.ks[b.ks.len() - 1]);
            if lo > hi {
                continue;
            }
            let mut grid: Vec<f64> =
                a.ks.iter()
                    .chain(b.ks.iter())
                    .copied()
                    .filter(|k| *k >= lo && *k <= hi)
                    .collect();
            grid.sort_by(f64::total_cmp);
            grid.dedup();
            for k in grid {
                let (wa, wb) = (
                    a.vol_at(k).powi(2) * a.expiry,
                    b.vol_at(k).powi(2) * b.expiry,
                );
                if wb < wa - W_TOL {
                    report.calendar.push(CalendarViolation {
                        log_moneyness: k,
                        expiry_short: a.expiry,
                        expiry_long: b.expiry,
                        total_variance_short: wa,
                        total_variance_long: wb,
                    });
                }
            }
        }

        let tol = 1e-10 * self.spot;
        for s in &self.smiles {
            let calls: Vec<f64> = s
                .strikes
                .iter()
                .zip(&s.vols)
                .map(|(&k, &v)| {
                    // Every node was validated against the supported domain in
                    // `new`; a NaN here (which can never trigger a violation)
                    // is only a defence against future changes.
                    OptionParams::builder()
                        .spot(self.spot)
                        .strike(k)
                        .rate(self.rate)
                        .dividend_yield(self.dividend_yield)
                        .volatility(v)
                        .expiry(s.expiry)
                        .build()
                        .map(|p| bsm::price(&p, OptionType::Call))
                        .unwrap_or(f64::NAN)
                })
                .collect();
            report.evaluation_errors += calls.iter().filter(|v| !v.is_finite()).count();
            butterfly_checks(
                &s.strikes,
                &calls,
                (-self.rate * s.expiry).exp(),
                s.expiry,
                tol,
                &mut report.butterfly,
            );
        }
        report
    }

    /// Looks for static arbitrage on a **sampled grid of the interpolated
    /// surface**, which also covers points between the quotes.
    ///
    /// Expiries are the quoted ones plus `extra_expiries_between` evenly
    /// spaced interior expiries between each adjacent pair. At each expiry,
    /// `strikes_per_expiry` strikes (at least 3) are spread uniformly in
    /// log-moneyness over the range quoted at **every** expiry, so no
    /// extrapolated points are examined. Calendar and butterfly/vertical
    /// conditions are tested exactly as in [`VolSurface::check_arbitrage`].
    ///
    /// A clean result means no violation was found at the sampled points. It
    /// does not prove the surface is arbitrage-free between them, in the
    /// wings, or in the flat extrapolation region. If the quoted strike
    /// ranges do not overlap, nothing is sampled and the report is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::surface::{ArbitrageScope, SurfaceSlice, VolSurface};
    /// let s = VolSurface::new(100.0, 0.02, 0.0, vec![
    ///     SurfaceSlice { expiry: 0.25, strikes: vec![90.0, 100.0, 110.0], vols: vec![0.26, 0.21, 0.23] },
    ///     SurfaceSlice { expiry: 1.00, strikes: vec![80.0, 100.0, 120.0], vols: vec![0.27, 0.22, 0.24] },
    /// ])?;
    /// let r = s.check_arbitrage_sampled(50, 3);
    /// assert!(matches!(r.scope, ArbitrageScope::SampledGrid { .. }));
    /// # Ok::<(), quantkit::QuantError>(())
    /// ```
    pub fn check_arbitrage_sampled(
        &self,
        strikes_per_expiry: usize,
        extra_expiries_between: usize,
    ) -> ArbitrageReport {
        let n = strikes_per_expiry.clamp(3, 10_000);
        let m = extra_expiries_between.min(1_000);
        let mut expiries = Vec::new();
        for (i, sl) in self.smiles.iter().enumerate() {
            expiries.push(sl.expiry);
            if let Some(next) = self.smiles.get(i + 1) {
                for j in 1..=m {
                    expiries
                        .push(sl.expiry + (next.expiry - sl.expiry) * j as f64 / (m + 1) as f64);
                }
            }
        }
        let mut report = ArbitrageReport {
            scope: ArbitrageScope::SampledGrid {
                strikes_per_expiry: n,
                expiries: expiries.len(),
            },
            ..ArbitrageReport::default()
        };
        let k_lo = self
            .smiles
            .iter()
            .map(|s| s.ks[0])
            .fold(f64::NEG_INFINITY, f64::max);
        let k_hi = self
            .smiles
            .iter()
            .map(|s| s.ks[s.ks.len() - 1])
            .fold(f64::INFINITY, f64::min);
        if k_lo >= k_hi {
            return report;
        }
        let ks: Vec<f64> = (0..n)
            .map(|i| k_lo + (k_hi - k_lo) * i as f64 / (n - 1) as f64)
            .collect();
        let tol = 1e-10 * self.spot;
        let mut previous: Option<(f64, Vec<f64>)> = None;
        for &t in &expiries {
            let log_fwd = self.log_forward(t);
            let strikes: Vec<f64> = ks.iter().map(|k| (log_fwd + k).exp()).collect();
            let vols: Vec<f64> = strikes
                .iter()
                .map(|&k| self.vol(k, t).unwrap_or(f64::NAN))
                .collect();
            let w: Vec<f64> = vols.iter().map(|v| v * v * t).collect();
            if let Some((t_prev, w_prev)) = &previous {
                for i in 0..n {
                    if w[i] < w_prev[i] - W_TOL {
                        report.calendar.push(CalendarViolation {
                            log_moneyness: ks[i],
                            expiry_short: *t_prev,
                            expiry_long: t,
                            total_variance_short: w_prev[i],
                            total_variance_long: w[i],
                        });
                    }
                }
            }
            let calls: Vec<f64> = strikes
                .iter()
                .zip(&vols)
                .map(|(&k, &v)| {
                    OptionParams::builder()
                        .spot(self.spot)
                        .strike(k)
                        .rate(self.rate)
                        .dividend_yield(self.dividend_yield)
                        .volatility(v)
                        .expiry(t)
                        .build()
                        .map(|p| bsm::price(&p, OptionType::Call))
                        .unwrap_or(f64::NAN)
                })
                .collect();
            report.evaluation_errors += calls.iter().filter(|v| !v.is_finite()).count();
            butterfly_checks(
                &strikes,
                &calls,
                (-self.rate * t).exp(),
                t,
                tol,
                &mut report.butterfly,
            );
            previous = Some((t, w));
        }
        report
    }
}

/// Tolerance on total variance for calendar comparisons.
const W_TOL: f64 = 1e-12;

/// Call-price no-arbitrage conditions across increasing `strikes` at one
/// expiry, with `df = exp(-r T)`:
///
/// * non-increasing: `C(K2) <= C(K1)`
/// * slope bound: `C(K1) - C(K2) <= df (K2 - K1)`
/// * convexity: `C(K2) <= lambda C(K1) + (1 - lambda) C(K3)`
///
/// NaN prices never trigger a violation.
fn butterfly_checks(
    strikes: &[f64],
    calls: &[f64],
    df: f64,
    expiry: f64,
    tol: f64,
    out: &mut Vec<ButterflyViolation>,
) {
    for j in 1..calls.len() {
        if calls[j] > calls[j - 1] + tol {
            out.push(ButterflyViolation {
                expiry,
                strike: strikes[j],
                excess: calls[j] - calls[j - 1],
            });
        }
        let slope_room = df * (strikes[j] - strikes[j - 1]);
        if calls[j - 1] - calls[j] > slope_room + tol {
            out.push(ButterflyViolation {
                expiry,
                strike: strikes[j],
                excess: calls[j - 1] - calls[j] - slope_room,
            });
        }
        if j + 1 < calls.len() {
            let (k0, k1, k2) = (strikes[j - 1], strikes[j], strikes[j + 1]);
            let lam = (k2 - k1) / (k2 - k0);
            let bound = lam * calls[j - 1] + (1.0 - lam) * calls[j + 1];
            if calls[j] > bound + tol {
                out.push(ButterflyViolation {
                    expiry,
                    strike: k1,
                    excess: calls[j] - bound,
                });
            }
        }
    }
}

/// Raw SVI parameters: total variance
/// `w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2))`.
///
/// # Examples
///
/// ```
/// use quantkit::surface::SviParams;
/// let svi = SviParams::new(0.04, 0.4, -0.4, 0.0, 0.1)?;
/// assert!(svi.total_variance(0.0) > 0.0);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SviParams {
    /// Overall variance level.
    pub a: f64,
    /// Slope of the wings (>= 0).
    pub b: f64,
    /// Skew, in `(-1, 1)`.
    pub rho: f64,
    /// Horizontal shift.
    pub m: f64,
    /// Smoothing at the vertex (> 0).
    pub sigma: f64,
}

impl SviParams {
    /// Validated constructor.
    ///
    /// # Errors
    ///
    /// Non-finite values, `b < 0`, `|rho| >= 1`, `sigma <= 0`, or a negative
    /// minimum total variance `a + b sigma sqrt(1 - rho^2)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::surface::SviParams;
    /// assert!(SviParams::new(0.04, 0.4, 1.0, 0.0, 0.1).is_err()); // |rho| = 1
    /// ```
    pub fn new(a: f64, b: f64, rho: f64, m: f64, sigma: f64) -> Result<Self> {
        for (n, v) in [("a", a), ("b", b), ("rho", rho), ("m", m), ("sigma", sigma)] {
            check_finite(n, v)?;
        }
        let bad = |name, reason| Err(QuantError::InvalidInput { name, reason });
        if b < 0.0 {
            return bad("b", "must be non-negative");
        }
        if rho.abs() >= 1.0 {
            return bad("rho", "must lie strictly between -1 and 1");
        }
        if sigma <= 0.0 {
            return bad("sigma", "must be strictly positive");
        }
        if a + b * sigma * (1.0 - rho * rho).sqrt() < 0.0 {
            return bad("a", "implies negative total variance");
        }
        Ok(Self {
            a,
            b,
            rho,
            m,
            sigma,
        })
    }

    /// Total variance `w(k)` at log-moneyness `k`.
    pub fn total_variance(&self, k: f64) -> f64 {
        let x = k - self.m;
        self.a + self.b * (self.rho * x + x.hypot(self.sigma))
    }

    /// Implied volatility at log-moneyness `k` and expiry `t`.
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::surface::SviParams;
    /// let svi = SviParams::new(0.04, 0.4, -0.4, 0.0, 0.1)?;
    /// let w = svi.total_variance(0.1);
    /// assert!((svi.vol(0.1, 2.0) - (w / 2.0).sqrt()).abs() < 1e-15);
    /// # Ok::<(), quantkit::QuantError>(())
    /// ```
    pub fn vol(&self, k: f64, t: f64) -> f64 {
        (self.total_variance(k) / t).sqrt()
    }

    /// Gatheral-Jacquier butterfly condition `g(k)`; the smile is free of
    /// butterfly arbitrage at `k` when `g(k) >= 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::surface::SviParams;
    /// let svi = SviParams::new(0.04, 0.4, -0.4, 0.0, 0.1)?;
    /// assert!(svi.butterfly_g(0.0).is_finite());
    /// # Ok::<(), quantkit::QuantError>(())
    /// ```
    pub fn butterfly_g(&self, k: f64) -> f64 {
        let x = k - self.m;
        let r = x.hypot(self.sigma);
        let w = self.total_variance(k);
        let w1 = self.b * (self.rho + x / r);
        let w2 = self.b * (self.sigma / r).powi(2) / r;
        let t = 1.0 - k * w1 / (2.0 * w);
        t * t - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2
    }

    /// Checks `g(k) >= 0` on `n` evenly spaced points in `[k_min, k_max]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use quantkit::surface::SviParams;
    /// let svi = SviParams::new(0.04, 0.4, -0.4, 0.0, 0.1)?;
    /// assert!(svi.is_butterfly_free(-1.0, 1.0, 201));
    /// # Ok::<(), quantkit::QuantError>(())
    /// ```
    pub fn is_butterfly_free(&self, k_min: f64, k_max: f64, n: usize) -> bool {
        let n = n.max(2);
        (0..n).all(|i| {
            let k = k_min + (k_max - k_min) * i as f64 / (n - 1) as f64;
            self.total_variance(k) > 0.0 && self.butterfly_g(k) >= 0.0
        })
    }
}

/// Result of [`fit_svi`].
///
/// # Examples
///
/// ```
/// use quantkit::surface::{fit_svi, SviParams};
/// let truth = SviParams::new(0.04, 0.4, -0.4, 0.0, 0.1)?;
/// let ks: Vec<f64> = (-10..=10).map(|i| f64::from(i) / 10.0).collect();
/// let ws: Vec<f64> = ks.iter().map(|&k| truth.total_variance(k)).collect();
/// let fit = fit_svi(&ks, &ws)?;
/// assert!(fit.rmse < 1e-8);
/// # Ok::<(), quantkit::QuantError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SviFit {
    /// Fitted parameters.
    pub params: SviParams,
    /// Root-mean-square error in total variance.
    pub rmse: f64,
    /// Largest absolute error in total variance over the input points.
    pub max_abs_residual: f64,
}

/// Least-squares fit of the raw SVI smile to `(log-moneyness, total variance)`
/// points, using the quasi-explicit method: for fixed `(m, sigma)` the
/// remaining three parameters solve a linear least-squares problem, and a
/// Nelder-Mead search runs over `(m, sigma)` from several starting points.
///
/// The fit enforces `b >= 0`, `|rho| < 1` and non-negative minimum variance,
/// but **not** the butterfly condition; check it with
/// [`SviParams::is_butterfly_free`].
///
/// # Errors
///
/// * [`QuantError::InvalidInput`] / [`QuantError::NonFinite`] if the slices
///   differ in length, contain fewer than 5 points, or contain non-finite or
///   negative values.
/// * [`QuantError::NoConvergence`] if no admissible parameters were found
///   (for example, perfectly flat data has no unique SVI representation).
pub fn fit_svi(ks: &[f64], ws: &[f64]) -> Result<SviFit> {
    if ks.len() != ws.len() {
        return Err(QuantError::InvalidInput {
            name: "ws",
            reason: "must have the same length as ks",
        });
    }
    if ks.len() < 5 {
        return Err(QuantError::InvalidInput {
            name: "ks",
            reason: "need at least 5 points to fit 5 parameters",
        });
    }
    if ks.iter().chain(ws).any(|x| !x.is_finite()) {
        return Err(QuantError::NonFinite { name: "ks/ws" });
    }
    if ws.iter().any(|&w| w < 0.0) {
        return Err(QuantError::InvalidInput {
            name: "ws",
            reason: "total variance cannot be negative",
        });
    }

    // Solves for (a, d = b*sigma*rho, c = b*sigma) at fixed (m, sigma).
    // Returns the coefficients, the sum of squared errors and a constraint
    // violation measure (0 when admissible).
    let linear = |m: f64, sigma: f64| -> Option<([f64; 3], f64, f64)> {
        let mut ata = [[0.0; 3]; 3];
        let mut atw = [0.0; 3];
        for (&k, &w) in ks.iter().zip(ws) {
            let y = (k - m) / sigma;
            let row = [1.0, y, (y * y + 1.0).sqrt()];
            for i in 0..3 {
                atw[i] += row[i] * w;
                for j in 0..3 {
                    ata[i][j] += row[i] * row[j];
                }
            }
        }
        let [a, d, c] = solve3(ata, atw)?;
        let sse: f64 = ks
            .iter()
            .zip(ws)
            .map(|(&k, &w)| {
                let y = (k - m) / sigma;
                (a + d * y + c * (y * y + 1.0).sqrt() - w).powi(2)
            })
            .sum();
        let violation = (-c).max(0.0)
            + (d.abs() - c).max(0.0)
            + (-(a + (c * c - d * d).max(0.0).sqrt())).max(0.0);
        Some(([a, d, c], sse, violation))
    };
    let objective = |p: &[f64; 2]| -> f64 {
        let (m, sigma) = (p[0], p[1].exp());
        match linear(m, sigma) {
            Some((_, sse, v)) => {
                if v == 0.0 {
                    sse
                } else {
                    sse + 1e-6 + 1e3 * v
                }
            }
            None => 1e10,
        }
    };

    let (k_at_min, _) = ks
        .iter()
        .zip(ws)
        .fold((ks[0], f64::INFINITY), |best, (&k, &w)| {
            if w < best.1 {
                (k, w)
            } else {
                best
            }
        });
    let mut best: Option<([f64; 2], f64)> = None;
    for &s0 in &[0.05_f64, 0.1, 0.3, 1.0] {
        for &m0 in &[k_at_min, 0.0] {
            let (x, f) = nelder_mead(objective, [m0, s0.ln()], 0.1, 4000);
            if best.map_or(true, |b| f < b.1) {
                best = Some((x, f));
            }
        }
    }
    let (x, _) = best.expect("at least one start");
    let (m, sigma) = (x[0], x[1].exp());
    let fail = |residual| QuantError::NoConvergence {
        iterations: 4000,
        residual,
    };
    let (coef, sse, violation) = linear(m, sigma).ok_or_else(|| fail(f64::INFINITY))?;
    if violation > 0.0 || coef[2] <= 0.0 {
        return Err(fail(violation));
    }
    let [a, d, c] = coef;
    let params = SviParams::new(
        a,
        c / sigma,
        (d / c).clamp(-0.999_999_999, 0.999_999_999),
        m,
        sigma,
    )
    .map_err(|_| fail(violation))?;
    let max_abs_residual = ks
        .iter()
        .zip(ws)
        .map(|(&k, &w)| (params.total_variance(k) - w).abs())
        .fold(0.0_f64, f64::max);
    Ok(SviFit {
        params,
        rmse: (sse / ks.len() as f64).sqrt(),
        max_abs_residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice(t: f64, strikes: &[f64], vols: &[f64]) -> SurfaceSlice {
        SurfaceSlice {
            expiry: t,
            strikes: strikes.to_vec(),
            vols: vols.to_vec(),
        }
    }

    #[test]
    fn flat_surface_is_flat_everywhere() {
        let s = VolSurface::new(
            100.0,
            0.03,
            0.01,
            vec![
                slice(0.25, &[80.0, 100.0, 120.0], &[0.2; 3]),
                slice(1.0, &[80.0, 100.0, 120.0], &[0.2; 3]),
            ],
        )
        .unwrap();
        for (k, t) in [(70.0, 0.1), (100.0, 0.25), (95.0, 0.6), (130.0, 5.0)] {
            assert!((s.vol(k, t).unwrap() - 0.2).abs() < 1e-12, "{k} {t}");
        }
        assert!(s.check_arbitrage().is_clean());
    }

    #[test]
    fn time_interpolation_is_linear_in_total_variance() {
        let s = VolSurface::new(
            100.0,
            0.0,
            0.0,
            vec![slice(0.5, &[100.0], &[0.20]), slice(1.5, &[100.0], &[0.30])],
        )
        .unwrap();
        // k = 0 for rate 0. w1 = 0.04*0.5 = 0.02, w2 = 0.09*1.5 = 0.135; at T = 1: w = 0.0775.
        let v = s.vol(100.0, 1.0).unwrap();
        assert!((v - 0.0775_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn smile_interpolation_hits_nodes_and_stays_between() {
        let s = VolSurface::new(
            100.0,
            0.0,
            0.0,
            vec![slice(1.0, &[90.0, 100.0, 110.0], &[0.25, 0.20, 0.23])],
        )
        .unwrap();
        // With zero rate the forward equals spot, so strikes are the nodes' own k.
        for (k, v) in [(90.0, 0.25), (100.0, 0.20), (110.0, 0.23)] {
            assert!((s.vol(k, 1.0).unwrap() - v).abs() < 1e-12);
        }
        let mid = s.vol(95.0, 1.0).unwrap();
        assert!(mid > 0.20 && mid < 0.25);
    }

    #[test]
    fn from_quotes_recovers_node_vols() {
        let smile = |k: f64| 0.2 + 0.1 * ((k / 100.0).ln()).powi(2);
        let mut quotes = Vec::new();
        for &t in &[0.25, 1.0] {
            for &k in &[85.0, 95.0, 100.0, 105.0, 115.0] {
                let kind = if k < 100.0 {
                    OptionType::Put
                } else {
                    OptionType::Call
                };
                let p = OptionParams::builder()
                    .spot(100.0)
                    .strike(k)
                    .rate(0.02)
                    .volatility(smile(k))
                    .expiry(t)
                    .build()
                    .unwrap();
                quotes.push(PriceQuote {
                    strike: k,
                    expiry: t,
                    kind,
                    price: bsm::price(&p, kind),
                });
            }
        }
        quotes.reverse(); // order must not matter
        let s = VolSurface::from_quotes(100.0, 0.02, 0.0, &quotes).unwrap();
        for &t in &[0.25, 1.0] {
            for &k in &[85.0, 95.0, 100.0, 105.0, 115.0] {
                assert!((s.vol(k, t).unwrap() - smile(k)).abs() < 1e-7, "{k} {t}");
            }
        }
    }

    #[test]
    fn from_quotes_propagates_bad_quote() {
        let q = PriceQuote {
            strike: 100.0,
            expiry: 1.0,
            kind: OptionType::Call,
            price: -1.0,
        };
        assert!(VolSurface::from_quotes(100.0, 0.0, 0.0, &[q]).is_err());
        assert!(VolSurface::from_quotes(100.0, 0.0, 0.0, &[]).is_err());
    }

    #[test]
    fn construction_validation() {
        let ok = || slice(1.0, &[90.0, 100.0], &[0.2, 0.2]);
        assert!(VolSurface::new(100.0, 0.0, 0.0, vec![ok()]).is_ok());
        assert!(VolSurface::new(0.0, 0.0, 0.0, vec![ok()]).is_err());
        assert!(VolSurface::new(100.0, f64::NAN, 0.0, vec![ok()]).is_err());
        assert!(VolSurface::new(100.0, 0.0, 0.0, vec![]).is_err());
        assert!(VolSurface::new(100.0, 0.0, 0.0, vec![slice(1.0, &[90.0], &[0.2, 0.3])]).is_err());
        assert!(VolSurface::new(100.0, 0.0, 0.0, vec![slice(1.0, &[90.0], &[-0.2])]).is_err());
        assert!(VolSurface::new(100.0, 0.0, 0.0, vec![slice(0.0, &[90.0], &[0.2])]).is_err());
        assert!(VolSurface::new(
            100.0,
            0.0,
            0.0,
            vec![slice(1.0, &[90.0, 90.0], &[0.2, 0.2])]
        )
        .is_err());
        assert!(VolSurface::new(100.0, 0.0, 0.0, vec![ok(), ok()]).is_err()); // same expiry twice
        let s = VolSurface::new(100.0, 0.0, 0.0, vec![ok()]).unwrap();
        assert!(s.vol(0.0, 1.0).is_err());
        assert!(s.vol(100.0, f64::NAN).is_err());
    }

    #[test]
    fn detects_calendar_arbitrage_only_where_present() {
        let bad = VolSurface::new(
            100.0,
            0.0,
            0.0,
            vec![slice(0.5, &[100.0], &[0.40]), slice(1.0, &[100.0], &[0.20])],
        )
        .unwrap();
        let r = bad.check_arbitrage();
        assert_eq!(r.calendar.len(), 1);
        assert!(r.calendar[0].total_variance_long < r.calendar[0].total_variance_short);
        let good = VolSurface::new(
            100.0,
            0.0,
            0.0,
            vec![slice(0.5, &[100.0], &[0.20]), slice(1.0, &[100.0], &[0.20])],
        )
        .unwrap();
        assert!(good.check_arbitrage().calendar.is_empty());
    }

    #[test]
    fn detects_butterfly_arbitrage() {
        // A vol spike at the middle strike makes the middle call too expensive.
        let s = VolSurface::new(
            100.0,
            0.0,
            0.0,
            vec![slice(1.0, &[90.0, 100.0, 110.0], &[0.20, 0.50, 0.20])],
        )
        .unwrap();
        assert!(!s.check_arbitrage().butterfly.is_empty());
        let smooth = VolSurface::new(
            100.0,
            0.0,
            0.0,
            vec![slice(1.0, &[90.0, 100.0, 110.0], &[0.22, 0.20, 0.21])],
        )
        .unwrap();
        assert!(smooth.check_arbitrage().butterfly.is_empty());
    }

    #[test]
    fn sampled_check_reports_scope_and_finds_violations() {
        let flat = VolSurface::new(
            100.0,
            0.02,
            0.0,
            vec![
                slice(0.25, &[80.0, 100.0, 120.0], &[0.2; 3]),
                slice(1.0, &[80.0, 100.0, 120.0], &[0.2; 3]),
            ],
        )
        .unwrap();
        let r = flat.check_arbitrage_sampled(40, 3);
        assert!(r.is_clean());
        assert_eq!(
            r.scope,
            ArbitrageScope::SampledGrid {
                strikes_per_expiry: 40,
                expiries: 5
            }
        );
        assert_eq!(flat.check_arbitrage().scope, ArbitrageScope::QuotedNodes);

        // A vol spike between the outer strikes breaks convexity on the
        // sampled interpolated grid as well.
        let spiked = VolSurface::new(
            100.0,
            0.0,
            0.0,
            vec![slice(1.0, &[90.0, 100.0, 110.0], &[0.20, 0.50, 0.20])],
        )
        .unwrap();
        assert!(!spiked.check_arbitrage_sampled(31, 0).butterfly.is_empty());

        // Calendar violation found between quotes' common range.
        let cal = VolSurface::new(
            100.0,
            0.0,
            0.0,
            vec![
                slice(0.5, &[90.0, 100.0, 110.0], &[0.40; 3]),
                slice(1.0, &[90.0, 100.0, 110.0], &[0.20; 3]),
            ],
        )
        .unwrap();
        assert!(!cal.check_arbitrage_sampled(11, 1).calendar.is_empty());

        // Disjoint quoted ranges: nothing sampled, empty (not "clean by proof").
        let disjoint = VolSurface::new(
            100.0,
            0.0,
            0.0,
            vec![
                slice(0.5, &[50.0, 60.0], &[0.2, 0.2]),
                slice(1.0, &[150.0, 160.0], &[0.2, 0.2]),
            ],
        )
        .unwrap();
        assert!(disjoint.check_arbitrage_sampled(11, 1).is_clean());
    }

    #[test]
    fn svi_validation_and_shape() {
        assert!(SviParams::new(0.04, -0.1, 0.0, 0.0, 0.1).is_err());
        assert!(SviParams::new(0.04, 0.4, 1.0, 0.0, 0.1).is_err());
        assert!(SviParams::new(0.04, 0.4, 0.0, 0.0, 0.0).is_err());
        assert!(SviParams::new(-1.0, 0.4, 0.0, 0.0, 0.1).is_err());
        assert!(SviParams::new(f64::NAN, 0.4, 0.0, 0.0, 0.1).is_err());
        // Negative rho gives a downward skew: variance higher on the left.
        let svi = SviParams::new(0.04, 0.4, -0.4, 0.0, 0.1).unwrap();
        assert!(svi.total_variance(-0.3) > svi.total_variance(0.3));
    }

    #[test]
    fn svi_butterfly_condition_distinguishes_good_and_bad() {
        let good = SviParams::new(0.04, 0.4, -0.4, 0.0, 0.1).unwrap();
        assert!(good.is_butterfly_free(-1.0, 1.0, 401));
        // Very steep wings with a tiny vertex radius violate g(k) >= 0.
        let bad = SviParams::new(0.0, 3.0, 0.9, 0.0, 0.01).unwrap();
        assert!(!bad.is_butterfly_free(-1.0, 1.0, 401));
    }

    #[test]
    fn svi_fit_recovers_noise_free_parameters() {
        let truth = SviParams::new(0.04, 0.4, -0.4, 0.05, 0.1).unwrap();
        let ks: Vec<f64> = (-20..=20).map(|i| f64::from(i) / 20.0).collect();
        let ws: Vec<f64> = ks.iter().map(|&k| truth.total_variance(k)).collect();
        let fit = fit_svi(&ks, &ws).unwrap();
        assert!(fit.rmse < 1e-9, "rmse {}", fit.rmse);
        let p = fit.params;
        assert!((p.a - truth.a).abs() < 1e-5, "a {}", p.a);
        assert!((p.b - truth.b).abs() < 1e-4, "b {}", p.b);
        assert!((p.rho - truth.rho).abs() < 1e-4, "rho {}", p.rho);
        assert!((p.m - truth.m).abs() < 1e-4, "m {}", p.m);
        assert!((p.sigma - truth.sigma).abs() < 1e-4, "sigma {}", p.sigma);
    }

    #[test]
    fn svi_fit_with_noise_has_small_error() {
        let truth = SviParams::new(0.03, 0.5, -0.5, 0.0, 0.15).unwrap();
        let mut rng = crate::mc::Rng::new(5);
        let ks: Vec<f64> = (-15..=15).map(|i| f64::from(i) / 15.0).collect();
        let ws: Vec<f64> = ks
            .iter()
            .map(|&k| truth.total_variance(k) * (1.0 + 0.002 * rng.next_normal()))
            .collect();
        let fit = fit_svi(&ks, &ws).unwrap();
        // The true parameters are feasible, so an optimal fit can never have a
        // larger error than the noise that was added.
        let noise_rmse = (ks
            .iter()
            .zip(&ws)
            .map(|(&k, &w)| (w - truth.total_variance(k)).powi(2))
            .sum::<f64>()
            / ks.len() as f64)
            .sqrt();
        assert!(
            fit.rmse <= noise_rmse * 1.0001,
            "rmse {} vs noise {noise_rmse}",
            fit.rmse
        );
    }

    #[test]
    fn svi_fit_rejects_bad_input() {
        let ks = [0.0, 0.1, 0.2, 0.3];
        assert!(fit_svi(&ks, &ks).is_err()); // too few points
        let ks5 = [0.0, 0.1, 0.2, 0.3, 0.4];
        assert!(fit_svi(&ks5, &[0.1; 4]).is_err());
        assert!(fit_svi(&ks5, &[0.1, 0.1, f64::NAN, 0.1, 0.1]).is_err());
        assert!(fit_svi(&ks5, &[0.1, -0.1, 0.1, 0.1, 0.1]).is_err());
    }
}
