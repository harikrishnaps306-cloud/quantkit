//! Python bindings for `quantkit`.
//!
//! Every function validates its inputs through the Rust crate and raises
//! `quantkit.QuantError` (a `ValueError` subclass) on bad input; nothing
//! panics across the FFI boundary. Batch functions take NumPy arrays (or
//! scalars, which broadcast) and release the GIL while they compute.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use quantkit as qk;
use qk::{ExerciseStyle, OptionParams, OptionType};

create_exception!(_quantkit, QuantError, PyValueError);

fn err(e: qk::QuantError) -> PyErr {
    QuantError::new_err(e.to_string())
}

fn kind(is_call: bool) -> OptionType {
    if is_call {
        OptionType::Call
    } else {
        OptionType::Put
    }
}

fn params(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    div_yield: f64,
) -> qk::Result<OptionParams> {
    OptionParams::builder()
        .spot(spot)
        .strike(strike)
        .rate(rate)
        .dividend_yield(div_yield)
        .volatility(vol)
        .expiry(expiry)
        .build()
}

/// A float or a 1-D float64 array; scalars broadcast against arrays.
#[derive(FromPyObject)]
enum Arg<'py> {
    Array(PyReadonlyArray1<'py, f64>),
    Scalar(f64),
}

impl Arg<'_> {
    fn len(&self) -> Option<usize> {
        match self {
            Arg::Array(a) => Some(a.as_array().len()),
            Arg::Scalar(_) => None,
        }
    }
    fn to_vec(&self, n: usize) -> Vec<f64> {
        match self {
            Arg::Array(a) => a.as_array().iter().copied().collect(),
            Arg::Scalar(x) => vec![*x; n],
        }
    }
}

/// Common length of the array arguments (1 if all are scalars).
fn common_len(args: &[&Arg<'_>]) -> PyResult<usize> {
    let mut n: Option<usize> = None;
    for a in args {
        if let Some(len) = a.len() {
            match n {
                Some(m) if m != len => {
                    return Err(PyValueError::new_err(format!(
                        "array arguments must have equal length, got {m} and {len}"
                    )))
                }
                _ => n = Some(len),
            }
        }
    }
    Ok(n.unwrap_or(1))
}

fn check_errors_mode(errors: &str) -> PyResult<bool> {
    match errors {
        "raise" => Ok(true),
        "nan" => Ok(false),
        _ => Err(PyValueError::new_err("errors must be 'raise' or 'nan'")),
    }
}

fn greeks_dict<'py>(py: Python<'py>, g: qk::greeks::Greeks) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("delta", g.delta)?;
    d.set_item("gamma", g.gamma)?;
    d.set_item("vega", g.vega)?;
    d.set_item("theta", g.theta)?;
    d.set_item("rho", g.rho)?;
    Ok(d)
}

/// Black-Scholes-Merton price of a European option.
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, expiry, is_call=true, div_yield=0.0))]
fn bs_price(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
    div_yield: f64,
) -> PyResult<f64> {
    let p = params(spot, strike, rate, vol, expiry, div_yield).map_err(err)?;
    Ok(qk::bsm::price(&p, kind(is_call)))
}

/// Analytic BSM Greeks as a dict: delta, gamma, vega (per 1.00 vol), theta (per year), rho (per 1.00 rate).
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, expiry, is_call=true, div_yield=0.0))]
fn bs_greeks<'py>(
    py: Python<'py>,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
    div_yield: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let p = params(spot, strike, rate, vol, expiry, div_yield).map_err(err)?;
    greeks_dict(py, qk::greeks::analytic(&p, kind(is_call)))
}

/// Implied volatility from a market price (Newton-Raphson with bisection fallback).
#[pyfunction]
#[pyo3(signature = (price, spot, strike, rate, expiry, is_call=true, div_yield=0.0))]
fn implied_vol(
    price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    expiry: f64,
    is_call: bool,
    div_yield: f64,
) -> PyResult<f64> {
    let c = OptionParams::builder()
        .spot(spot)
        .strike(strike)
        .rate(rate)
        .dividend_yield(div_yield)
        .expiry(expiry)
        .build_contract()
        .map_err(err)?;
    qk::implied_vol::solve(&c, kind(is_call), price).map_err(err)
}

/// CRR binomial price (American by default).
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, expiry, is_call=true, div_yield=0.0, american=true, steps=200))]
#[allow(clippy::too_many_arguments)]
fn binomial_price(
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
    div_yield: f64,
    american: bool,
    steps: usize,
) -> PyResult<f64> {
    let p = params(spot, strike, rate, vol, expiry, div_yield).map_err(err)?;
    let style = if american {
        ExerciseStyle::American
    } else {
        ExerciseStyle::European
    };
    qk::binomial::price(&p, kind(is_call), style, steps).map_err(err)
}

/// Monte Carlo European price. Returns (price, standard_error).
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, expiry, is_call=true, div_yield=0.0, paths=100_000, seed=42, antithetic=true))]
#[allow(clippy::too_many_arguments)]
fn mc_european(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
    div_yield: f64,
    paths: usize,
    seed: u64,
    antithetic: bool,
) -> PyResult<(f64, f64)> {
    let p = params(spot, strike, rate, vol, expiry, div_yield).map_err(err)?;
    let cfg = qk::mc::McConfig {
        paths,
        seed,
        antithetic,
    };
    let r = py
        .detach(|| qk::mc::european(&p, kind(is_call), cfg))
        .map_err(err)?;
    Ok((r.price, r.std_error))
}

/// Longstaff-Schwartz Monte Carlo American price. Returns (price, standard_error).
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, expiry, is_call=false, div_yield=0.0, steps=50, paths=100_000, seed=42, antithetic=true))]
#[allow(clippy::too_many_arguments)]
fn mc_american(
    py: Python<'_>,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
    div_yield: f64,
    steps: usize,
    paths: usize,
    seed: u64,
    antithetic: bool,
) -> PyResult<(f64, f64)> {
    let p = params(spot, strike, rate, vol, expiry, div_yield).map_err(err)?;
    let cfg = qk::mc::McConfig {
        paths,
        seed,
        antithetic,
    };
    let r = py
        .detach(|| qk::mc::american_lsm(&p, kind(is_call), steps, cfg))
        .map_err(err)?;
    Ok((r.price, r.std_error))
}

/// Vectorised BSM prices. Scalars broadcast. `errors='raise'` (default) raises on the
/// first invalid row; `errors='nan'` returns NaN for invalid rows.
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, expiry, is_call=true, div_yield=None, errors="raise"))]
#[allow(clippy::too_many_arguments)]
fn bs_price_batch<'py>(
    py: Python<'py>,
    spot: Arg<'py>,
    strike: Arg<'py>,
    rate: Arg<'py>,
    vol: Arg<'py>,
    expiry: Arg<'py>,
    is_call: bool,
    div_yield: Option<Arg<'py>>,
    errors: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let raise = check_errors_mode(errors)?;
    let div_yield = div_yield.unwrap_or(Arg::Scalar(0.0));
    let n = common_len(&[&spot, &strike, &rate, &vol, &expiry, &div_yield])?;
    let cols = [
        spot.to_vec(n),
        strike.to_vec(n),
        rate.to_vec(n),
        vol.to_vec(n),
        expiry.to_vec(n),
        div_yield.to_vec(n),
    ];
    let built: Vec<qk::Result<OptionParams>> = (0..n)
        .map(|i| {
            params(
                cols[0][i], cols[1][i], cols[2][i], cols[3][i], cols[4][i], cols[5][i],
            )
        })
        .collect();
    let (valid, index) = split_valid(built, raise, n)?;
    let prices = py.detach(|| qk::batch::price(&valid, kind(is_call)));
    Ok(scatter(prices, &index, n).into_pyarray(py))
}

/// Vectorised analytic Greeks. Returns a dict of float64 arrays.
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, expiry, is_call=true, div_yield=None, errors="raise"))]
#[allow(clippy::too_many_arguments)]
fn bs_greeks_batch<'py>(
    py: Python<'py>,
    spot: Arg<'py>,
    strike: Arg<'py>,
    rate: Arg<'py>,
    vol: Arg<'py>,
    expiry: Arg<'py>,
    is_call: bool,
    div_yield: Option<Arg<'py>>,
    errors: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let raise = check_errors_mode(errors)?;
    let div_yield = div_yield.unwrap_or(Arg::Scalar(0.0));
    let n = common_len(&[&spot, &strike, &rate, &vol, &expiry, &div_yield])?;
    let cols = [
        spot.to_vec(n),
        strike.to_vec(n),
        rate.to_vec(n),
        vol.to_vec(n),
        expiry.to_vec(n),
        div_yield.to_vec(n),
    ];
    let built: Vec<qk::Result<OptionParams>> = (0..n)
        .map(|i| {
            params(
                cols[0][i], cols[1][i], cols[2][i], cols[3][i], cols[4][i], cols[5][i],
            )
        })
        .collect();
    let (valid, index) = split_valid(built, raise, n)?;
    let gs = py.detach(|| qk::batch::greeks(&valid, kind(is_call)));
    let pick = |f: fn(&qk::greeks::Greeks) -> f64| -> Vec<f64> {
        scatter(gs.iter().map(f).collect(), &index, n)
    };
    let d = PyDict::new(py);
    d.set_item("delta", pick(|g| g.delta).into_pyarray(py))?;
    d.set_item("gamma", pick(|g| g.gamma).into_pyarray(py))?;
    d.set_item("vega", pick(|g| g.vega).into_pyarray(py))?;
    d.set_item("theta", pick(|g| g.theta).into_pyarray(py))?;
    d.set_item("rho", pick(|g| g.rho).into_pyarray(py))?;
    Ok(d)
}

/// Vectorised implied volatility. Unsolvable rows give NaN with `errors='nan'`.
#[pyfunction]
#[pyo3(signature = (price, spot, strike, rate, expiry, is_call=true, div_yield=None, errors="raise"))]
#[allow(clippy::too_many_arguments)]
fn implied_vol_batch<'py>(
    py: Python<'py>,
    price: Arg<'py>,
    spot: Arg<'py>,
    strike: Arg<'py>,
    rate: Arg<'py>,
    expiry: Arg<'py>,
    is_call: bool,
    div_yield: Option<Arg<'py>>,
    errors: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let raise = check_errors_mode(errors)?;
    let div_yield = div_yield.unwrap_or(Arg::Scalar(0.0));
    let n = common_len(&[&price, &spot, &strike, &rate, &expiry, &div_yield])?;
    let cols = [
        price.to_vec(n),
        spot.to_vec(n),
        strike.to_vec(n),
        rate.to_vec(n),
        expiry.to_vec(n),
        div_yield.to_vec(n),
    ];
    let k = kind(is_call);
    let mut quotes = Vec::with_capacity(n);
    let mut idx = Vec::with_capacity(n);
    for i in 0..n {
        let c = OptionParams::builder()
            .spot(cols[1][i])
            .strike(cols[2][i])
            .rate(cols[3][i])
            .dividend_yield(cols[5][i])
            .expiry(cols[4][i])
            .build_contract();
        match c {
            Ok(contract) => {
                quotes.push(qk::batch::IvQuote {
                    contract,
                    kind: k,
                    price: cols[0][i],
                });
                idx.push(i);
            }
            Err(e) if raise => return Err(err(e)),
            Err(_) => {}
        }
    }
    let solved = py.detach(|| qk::batch::implied_vol(&quotes));
    let mut out = vec![f64::NAN; n];
    for (slot, r) in idx.into_iter().zip(solved) {
        match r {
            Ok(v) => out[slot] = v,
            Err(e) if raise => return Err(err(e)),
            Err(_) => {}
        }
    }
    Ok(out.into_pyarray(py))
}

/// Separates valid rows from invalid ones. With `raise`, the first error is raised.
fn split_valid(
    built: Vec<qk::Result<OptionParams>>,
    raise: bool,
    n: usize,
) -> PyResult<(Vec<OptionParams>, Vec<usize>)> {
    let mut valid = Vec::with_capacity(n);
    let mut index = Vec::with_capacity(n);
    for (i, r) in built.into_iter().enumerate() {
        match r {
            Ok(p) => {
                valid.push(p);
                index.push(i);
            }
            Err(e) if raise => {
                return Err(QuantError::new_err(format!("row {i}: {e}")));
            }
            Err(_) => {}
        }
    }
    Ok((valid, index))
}

/// Places `values` at `index` in a NaN-filled vector of length `n`.
fn scatter(values: Vec<f64>, index: &[usize], n: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; n];
    for (v, &i) in values.into_iter().zip(index) {
        out[i] = v;
    }
    out
}

macro_rules! risk_fn {
    ($name:ident, $doc:literal, $call:path, $( $arg:ident : $ty:ty ),*) => {
        #[doc = $doc]
        #[pyfunction]
        fn $name(returns: PyReadonlyArray1<'_, f64>, $( $arg: $ty ),*) -> PyResult<f64> {
            let r = returns.as_array();
            let slice: Vec<f64>;
            let s = match r.as_slice() {
                Some(s) => s,
                None => {
                    slice = r.iter().copied().collect();
                    &slice
                }
            };
            $call(s, $( $arg ),*).map_err(err)
        }
    };
}

risk_fn!(historical_var, "Historical VaR as a positive loss fraction.", qk::risk::historical_var, confidence: f64);
risk_fn!(historical_cvar, "Historical CVaR (expected shortfall) as a positive loss fraction.", qk::risk::historical_cvar, confidence: f64);
risk_fn!(parametric_var, "Normal (variance-covariance) VaR as a positive loss fraction.", qk::risk::parametric_var, confidence: f64);
risk_fn!(sharpe_ratio, "Annualised Sharpe ratio.", qk::risk::sharpe_ratio, risk_free_per_period: f64, periods_per_year: f64);
risk_fn!(sortino_ratio, "Annualised Sortino ratio.", qk::risk::sortino_ratio, target_per_period: f64, periods_per_year: f64);
risk_fn!(max_drawdown, "Maximum drawdown of the compounded equity curve, as a fraction.", qk::risk::max_drawdown,);
risk_fn!(annualized_volatility, "Annualised volatility (sample standard deviation).", qk::risk::annualized_volatility, periods_per_year: f64);

/// Fits a raw SVI smile to (log_moneyness, total_variance) points. Returns a dict with
/// a, b, rho, m, sigma and rmse.
#[pyfunction]
fn svi_fit<'py>(
    py: Python<'py>,
    log_moneyness: PyReadonlyArray1<'py, f64>,
    total_variance: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let ks: Vec<f64> = log_moneyness.as_array().iter().copied().collect();
    let ws: Vec<f64> = total_variance.as_array().iter().copied().collect();
    let fit = py
        .detach(|| qk::surface::fit_svi(&ks, &ws))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("a", fit.params.a)?;
    d.set_item("b", fit.params.b)?;
    d.set_item("rho", fit.params.rho)?;
    d.set_item("m", fit.params.m)?;
    d.set_item("sigma", fit.params.sigma)?;
    d.set_item("rmse", fit.rmse)?;
    Ok(d)
}

/// Implied-volatility surface built from smiles, with interpolation and arbitrage checks.
#[pyclass(name = "VolSurface", module = "quantkit")]
struct PyVolSurface {
    inner: qk::surface::VolSurface,
}

#[pymethods]
impl PyVolSurface {
    /// `slices` is a list of `(expiry, strikes, vols)` tuples with increasing expiry.
    #[new]
    #[pyo3(signature = (spot, slices, rate=0.0, div_yield=0.0))]
    fn new(
        spot: f64,
        slices: Vec<(f64, Vec<f64>, Vec<f64>)>,
        rate: f64,
        div_yield: f64,
    ) -> PyResult<Self> {
        let slices = slices
            .into_iter()
            .map(|(expiry, strikes, vols)| qk::surface::SurfaceSlice {
                expiry,
                strikes,
                vols,
            })
            .collect();
        let inner = qk::surface::VolSurface::new(spot, rate, div_yield, slices).map_err(err)?;
        Ok(Self { inner })
    }

    /// Interpolated implied volatility at (strike, expiry).
    fn vol(&self, strike: f64, expiry: f64) -> PyResult<f64> {
        self.inner.vol(strike, expiry).map_err(err)
    }

    /// Returns `{'calendar': n, 'butterfly': n}`, the number of static-arbitrage violations found at the quoted nodes.
    fn check_arbitrage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let r = self.inner.check_arbitrage();
        let d = PyDict::new(py);
        d.set_item("calendar", r.calendar.len())?;
        d.set_item("butterfly", r.butterfly.len())?;
        Ok(d)
    }
}

#[pymodule]
fn _quantkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("QuantError", m.py().get_type::<QuantError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyVolSurface>()?;
    m.add_function(wrap_pyfunction!(bs_price, m)?)?;
    m.add_function(wrap_pyfunction!(bs_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(implied_vol, m)?)?;
    m.add_function(wrap_pyfunction!(binomial_price, m)?)?;
    m.add_function(wrap_pyfunction!(mc_european, m)?)?;
    m.add_function(wrap_pyfunction!(mc_american, m)?)?;
    m.add_function(wrap_pyfunction!(bs_price_batch, m)?)?;
    m.add_function(wrap_pyfunction!(bs_greeks_batch, m)?)?;
    m.add_function(wrap_pyfunction!(implied_vol_batch, m)?)?;
    m.add_function(wrap_pyfunction!(historical_var, m)?)?;
    m.add_function(wrap_pyfunction!(historical_cvar, m)?)?;
    m.add_function(wrap_pyfunction!(parametric_var, m)?)?;
    m.add_function(wrap_pyfunction!(sharpe_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(sortino_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(max_drawdown, m)?)?;
    m.add_function(wrap_pyfunction!(annualized_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(svi_fit, m)?)?;
    Ok(())
}
