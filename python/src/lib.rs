//! Python bindings for `quantkit`.
//!
//! Every function validates its inputs through the Rust crate and raises
//! `quantkit.QuantError` (a `ValueError` subclass) on bad input; nothing
//! panics across the FFI boundary.
//!
//! Batch functions take NumPy arrays of any dimension (all array arguments
//! must have the **same shape**; scalars broadcast) and release the GIL while
//! computing. If every argument is a scalar, a scalar is returned. The
//! `*_detailed` variants return one status code and diagnostics per row so a
//! `NaN` never has to be guessed at.

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList};
use pyo3::IntoPyObjectExt;

use qk::detailed::{RawOption, RawQuote};
use qk::{ExerciseStyle, OptionParams, OptionType, StatusCode};
use quantkit as qk;

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

/// A float or an n-dimensional float64 array; scalars broadcast.
#[derive(FromPyObject)]
enum Arg<'py> {
    Array(PyReadonlyArrayDyn<'py, f64>),
    Scalar(f64),
}

/// Flattened columns plus the common output shape (`None` = all scalars).
struct Cols {
    n: usize,
    shape: Option<Vec<usize>>,
    cols: Vec<Vec<f64>>,
}

fn gather(args: &[&Arg<'_>]) -> PyResult<Cols> {
    let mut shape: Option<Vec<usize>> = None;
    for a in args {
        if let Arg::Array(arr) = a {
            let s = arr.as_array().shape().to_vec();
            match &shape {
                Some(prev) if *prev != s => {
                    return Err(PyValueError::new_err(format!(
                        "array arguments must have the same shape (got {prev:?} and {s:?}); \
                         only scalars broadcast"
                    )))
                }
                _ => shape = Some(s),
            }
        }
    }
    let n = shape.as_ref().map_or(1, |s| s.iter().product());
    let cols = args
        .iter()
        .map(|a| match a {
            Arg::Array(arr) => arr.as_array().iter().copied().collect(),
            Arg::Scalar(x) => vec![*x; n],
        })
        .collect();
    Ok(Cols { n, shape, cols })
}

/// Float array (or float for all-scalar input) in the input shape.
fn float_out<'py>(
    py: Python<'py>,
    v: Vec<f64>,
    shape: &Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyAny>> {
    match shape {
        None => Ok(PyFloat::new(py, v[0]).into_any()),
        Some(s) => Ok(v.into_pyarray(py).reshape(s.clone())?.into_any()),
    }
}

/// Unsigned-integer array (or Python int for all-scalar input) in the input shape.
fn uint_out<'py, T>(
    py: Python<'py>,
    v: Vec<T>,
    shape: &Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: numpy::Element + Copy + pyo3::IntoPyObject<'py>,
{
    match shape {
        None => v[0].into_bound_py_any(py),
        Some(s) => Ok(v.into_pyarray(py).reshape(s.clone())?.into_any()),
    }
}

fn messages<'py>(py: Python<'py>, errs: &[Option<qk::QuantError>]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, errs.iter().map(|e| e.as_ref().map(|e| e.to_string())))
}

fn check_errors_mode(errors: &str) -> PyResult<bool> {
    match errors {
        "raise" => Ok(true),
        "nan" => Ok(false),
        _ => Err(PyValueError::new_err("errors must be 'raise' or 'nan'")),
    }
}

/// Raises the first failed row (when `raise`), naming its index.
fn raise_first(
    raise: bool,
    statuses: &[StatusCode],
    errs: &[Option<qk::QuantError>],
) -> PyResult<()> {
    if raise {
        if let Some(i) = statuses.iter().position(|s| *s != StatusCode::Ok) {
            let msg = errs[i].as_ref().map(|e| e.to_string()).unwrap_or_default();
            return Err(QuantError::new_err(format!("row {i}: {msg}")));
        }
    }
    Ok(())
}

/// Raises the first failed row of a column result (when `raise`).
fn raise_columns(raise: bool, errors: &[(usize, qk::QuantError)]) -> PyResult<()> {
    if raise {
        if let Some((i, e)) = errors.first() {
            return Err(QuantError::new_err(format!("row {i}: {e}")));
        }
    }
    Ok(())
}

/// Per-row message list (None for successful rows) from sparse errors.
fn sparse_messages<'py>(
    py: Python<'py>,
    n: usize,
    errors: &[(usize, qk::QuantError)],
) -> PyResult<Bound<'py, PyList>> {
    let mut msgs: Vec<Option<String>> = vec![None; n];
    for (i, e) in errors {
        msgs[*i] = Some(e.to_string());
    }
    PyList::new(py, msgs)
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
#[allow(clippy::too_many_arguments)]
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

/// Monte Carlo European price. Returns (price, standard_error). Results are
/// reproducible for a given seed and independent of thread count.
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

/// Longstaff-Schwartz American price with the exercise policy trained on one
/// set of paths and evaluated on independent paths. Returns (price, standard_error).
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

/// Longstaff-Schwartz with separate training and evaluation sizes/seeds.
/// `price`/`std_error` are the value of the fixed trained policy (including the
/// time-zero decision, taken from the training paths) and the sampling error of
/// its evaluation mean given that policy. Also returns the raw continuation value,
/// the optimistic in-sample estimate, and observation counts (`*_paths` count
/// simulated paths; `*_samples` count independent observations, antithetic pairs
/// counting once).
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, expiry, is_call=false, div_yield=0.0, steps=50, train_paths=50_000, eval_paths=100_000, train_seed=1, eval_seed=2, antithetic=true))]
#[allow(clippy::too_many_arguments)]
fn mc_american_detailed<'py>(
    py: Python<'py>,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
    div_yield: f64,
    steps: usize,
    train_paths: usize,
    eval_paths: usize,
    train_seed: u64,
    eval_seed: u64,
    antithetic: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let p = params(spot, strike, rate, vol, expiry, div_yield).map_err(err)?;
    let train = qk::mc::McConfig {
        paths: train_paths,
        seed: train_seed,
        antithetic,
    };
    let eval = qk::mc::McConfig {
        paths: eval_paths,
        seed: eval_seed,
        antithetic,
    };
    let r = py
        .detach(|| qk::mc::american_lsm_detailed(&p, kind(is_call), steps, train, eval))
        .map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("price", r.price)?;
    d.set_item("std_error", r.std_error)?;
    d.set_item("exercise_now", r.exercise_now)?;
    d.set_item("immediate_exercise_value", r.immediate_exercise_value)?;
    d.set_item("continuation_value", r.continuation_value)?;
    d.set_item("continuation_std_error", r.continuation_std_error)?;
    d.set_item("in_sample_price", r.in_sample_price)?;
    d.set_item("in_sample_std_error", r.in_sample_std_error)?;
    d.set_item("train_paths", r.train_paths)?;
    d.set_item("train_samples", r.train_samples)?;
    d.set_item("eval_paths", r.eval_paths)?;
    d.set_item("eval_samples", r.eval_samples)?;
    d.set_item("fitted_dates", r.fitted_dates)?;
    d.set_item("regression_failures", r.regression_failures)?;
    Ok(d)
}

fn raw_rows(c: &Cols) -> Vec<RawOption> {
    // Column order: spot, strike, rate, vol, expiry, div_yield.
    (0..c.n)
        .map(|i| RawOption {
            spot: c.cols[0][i],
            strike: c.cols[1][i],
            rate: c.cols[2][i],
            volatility: c.cols[3][i],
            expiry: c.cols[4][i],
            dividend_yield: c.cols[5][i],
        })
        .collect()
}

fn raw_quotes(c: &Cols) -> Vec<RawQuote> {
    // Column order: price, spot, strike, rate, expiry, div_yield.
    (0..c.n)
        .map(|i| RawQuote {
            price: c.cols[0][i],
            spot: c.cols[1][i],
            strike: c.cols[2][i],
            rate: c.cols[3][i],
            expiry: c.cols[4][i],
            dividend_yield: c.cols[5][i],
        })
        .collect()
}

fn codes(statuses: &[StatusCode]) -> Vec<u8> {
    statuses.iter().map(|s| *s as u8).collect()
}

/// Vectorised BSM prices. Array arguments must share a shape; scalars broadcast;
/// all-scalar input returns a float. `errors='raise'` (default) raises on the
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
) -> PyResult<Bound<'py, PyAny>> {
    let raise = check_errors_mode(errors)?;
    let div_yield = div_yield.unwrap_or(Arg::Scalar(0.0));
    let c = gather(&[&spot, &strike, &rate, &vol, &expiry, &div_yield])?;
    let out = py
        .detach(|| {
            qk::detailed::price_columns(
                &c.cols[0],
                &c.cols[1],
                &c.cols[2],
                &c.cols[5],
                &c.cols[3],
                &c.cols[4],
                kind(is_call),
            )
        })
        .map_err(err)?;
    raise_columns(raise, &out.errors)?;
    let [price] = out.values;
    float_out(py, price, &c.shape)
}

/// Like `bs_price_batch`, returning a dict with `price`, `status` (uint8 codes, see
/// `quantkit.STATUS`) and `messages` (flattened list; None where the row succeeded).
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, expiry, is_call=true, div_yield=None))]
#[allow(clippy::too_many_arguments)]
fn bs_price_detailed<'py>(
    py: Python<'py>,
    spot: Arg<'py>,
    strike: Arg<'py>,
    rate: Arg<'py>,
    vol: Arg<'py>,
    expiry: Arg<'py>,
    is_call: bool,
    div_yield: Option<Arg<'py>>,
) -> PyResult<Bound<'py, PyDict>> {
    let div_yield = div_yield.unwrap_or(Arg::Scalar(0.0));
    let c = gather(&[&spot, &strike, &rate, &vol, &expiry, &div_yield])?;
    let out = py
        .detach(|| {
            qk::detailed::price_columns(
                &c.cols[0],
                &c.cols[1],
                &c.cols[2],
                &c.cols[5],
                &c.cols[3],
                &c.cols[4],
                kind(is_call),
            )
        })
        .map_err(err)?;
    let d = PyDict::new(py);
    let [price] = out.values;
    d.set_item("price", float_out(py, price, &c.shape)?)?;
    d.set_item("status", uint_out(py, codes(&out.status), &c.shape)?)?;
    d.set_item("messages", sparse_messages(py, c.n, &out.errors)?)?;
    Ok(d)
}

/// Vectorised analytic Greeks. Returns a dict of float64 arrays (floats for all-scalar input).
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
    let c = gather(&[&spot, &strike, &rate, &vol, &expiry, &div_yield])?;
    let out = py
        .detach(|| {
            qk::detailed::greeks_columns(
                &c.cols[0],
                &c.cols[1],
                &c.cols[2],
                &c.cols[5],
                &c.cols[3],
                &c.cols[4],
                kind(is_call),
            )
        })
        .map_err(err)?;
    raise_columns(raise, &out.errors)?;
    let d = PyDict::new(py);
    let names = ["delta", "gamma", "vega", "theta", "rho"];
    for (name, col) in names.into_iter().zip(out.values) {
        d.set_item(name, float_out(py, col, &c.shape)?)?;
    }
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
) -> PyResult<Bound<'py, PyAny>> {
    let raise = check_errors_mode(errors)?;
    let div_yield = div_yield.unwrap_or(Arg::Scalar(0.0));
    let c = gather(&[&price, &spot, &strike, &rate, &expiry, &div_yield])?;
    let quotes = raw_quotes(&c);
    let cfg = qk::implied_vol::SolverConfig::default();
    let out = py.detach(|| qk::detailed::implied_vol(&quotes, kind(is_call), cfg));
    let st: Vec<StatusCode> = out.iter().map(|r| r.status).collect();
    let er: Vec<_> = out.iter().map(|r| r.error.clone()).collect();
    raise_first(raise, &st, &er)?;
    float_out(py, out.iter().map(|r| r.vol).collect(), &c.shape)
}

/// Implied volatility with per-row diagnostics: `vol`, `status`, `residual` (absolute
/// price error at `vol`, or the last residual of a non-converged row), `iterations`,
/// `newton_steps`, `vega`, `elasticity` (d ln price / d ln vol: a relative price error e becomes a relative vol error of about e / elasticity) and `messages`.
#[pyfunction]
#[pyo3(signature = (price, spot, strike, rate, expiry, is_call=true, div_yield=None, tolerance=1e-12, max_iterations=100))]
#[allow(clippy::too_many_arguments)]
fn implied_vol_detailed<'py>(
    py: Python<'py>,
    price: Arg<'py>,
    spot: Arg<'py>,
    strike: Arg<'py>,
    rate: Arg<'py>,
    expiry: Arg<'py>,
    is_call: bool,
    div_yield: Option<Arg<'py>>,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let div_yield = div_yield.unwrap_or(Arg::Scalar(0.0));
    let c = gather(&[&price, &spot, &strike, &rate, &expiry, &div_yield])?;
    let quotes = raw_quotes(&c);
    let cfg = qk::implied_vol::SolverConfig {
        tolerance,
        max_iterations,
    };
    let out = py.detach(|| qk::detailed::implied_vol(&quotes, kind(is_call), cfg));
    let st: Vec<StatusCode> = out.iter().map(|r| r.status).collect();
    let er: Vec<_> = out.iter().map(|r| r.error.clone()).collect();
    let d = PyDict::new(py);
    d.set_item(
        "vol",
        float_out(py, out.iter().map(|r| r.vol).collect(), &c.shape)?,
    )?;
    d.set_item("status", uint_out(py, codes(&st), &c.shape)?)?;
    d.set_item(
        "residual",
        float_out(py, out.iter().map(|r| r.residual).collect(), &c.shape)?,
    )?;
    d.set_item(
        "iterations",
        uint_out(
            py,
            out.iter().map(|r| r.iterations as u64).collect(),
            &c.shape,
        )?,
    )?;
    d.set_item(
        "newton_steps",
        uint_out(
            py,
            out.iter().map(|r| r.newton_steps as u64).collect(),
            &c.shape,
        )?,
    )?;
    d.set_item(
        "vega",
        float_out(py, out.iter().map(|r| r.vega).collect(), &c.shape)?,
    )?;
    d.set_item(
        "elasticity",
        float_out(py, out.iter().map(|r| r.elasticity).collect(), &c.shape)?,
    )?;
    d.set_item("messages", messages(py, &er)?)?;
    Ok(d)
}

/// Monte Carlo European prices for many rows with per-row `std_error`, `status`, `seed`
/// and `messages`. Row i uses a seed derived from `seed` and i, so rows are independent
/// and the result does not depend on thread count.
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, vol, expiry, is_call=true, div_yield=None, paths=100_000, seed=42, antithetic=true))]
#[allow(clippy::too_many_arguments)]
fn mc_european_detailed<'py>(
    py: Python<'py>,
    spot: Arg<'py>,
    strike: Arg<'py>,
    rate: Arg<'py>,
    vol: Arg<'py>,
    expiry: Arg<'py>,
    is_call: bool,
    div_yield: Option<Arg<'py>>,
    paths: usize,
    seed: u64,
    antithetic: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let div_yield = div_yield.unwrap_or(Arg::Scalar(0.0));
    let c = gather(&[&spot, &strike, &rate, &vol, &expiry, &div_yield])?;
    let rows = raw_rows(&c);
    let cfg = qk::mc::McConfig {
        paths,
        seed,
        antithetic,
    };
    let out = py.detach(|| qk::detailed::mc_european(&rows, kind(is_call), cfg));
    let st: Vec<StatusCode> = out.iter().map(|r| r.status).collect();
    let er: Vec<_> = out.iter().map(|r| r.error.clone()).collect();
    let d = PyDict::new(py);
    d.set_item(
        "price",
        float_out(py, out.iter().map(|r| r.price).collect(), &c.shape)?,
    )?;
    d.set_item(
        "std_error",
        float_out(py, out.iter().map(|r| r.std_error).collect(), &c.shape)?,
    )?;
    d.set_item("status", uint_out(py, codes(&st), &c.shape)?)?;
    d.set_item(
        "seed",
        uint_out(py, out.iter().map(|r| r.seed).collect(), &c.shape)?,
    )?;
    d.set_item("messages", messages(py, &er)?)?;
    Ok(d)
}

macro_rules! risk_fn {
    ($name:ident, $doc:literal, $call:path, $( $arg:ident : $ty:ty ),*) => {
        #[doc = $doc]
        #[pyfunction]
        fn $name(returns: PyReadonlyArrayDyn<'_, f64>, $( $arg: $ty ),*) -> PyResult<f64> {
            let v: Vec<f64> = returns.as_array().iter().copied().collect();
            $call(&v, $( $arg ),*).map_err(err)
        }
    };
}

risk_fn!(historical_var, "Historical VaR as a positive loss fraction.", qk::risk::historical_var, confidence: f64);
risk_fn!(historical_cvar, "Historical CVaR (expected shortfall) as a positive loss fraction.", qk::risk::historical_cvar, confidence: f64);
risk_fn!(parametric_var, "Normal (variance-covariance) VaR as a positive loss fraction.", qk::risk::parametric_var, confidence: f64);
risk_fn!(sharpe_ratio, "Annualised Sharpe ratio (per-period risk-free rate).", qk::risk::sharpe_ratio, risk_free_per_period: f64, periods_per_year: f64);
risk_fn!(sortino_ratio, "Annualised Sortino ratio (per-period target).", qk::risk::sortino_ratio, target_per_period: f64, periods_per_year: f64);
risk_fn!(
    max_drawdown,
    "Maximum drawdown of the compounded equity curve, as a fraction.",
    qk::risk::max_drawdown,
);
risk_fn!(annualized_volatility, "Annualised volatility (sample standard deviation).", qk::risk::annualized_volatility, periods_per_year: f64);

/// Fits a raw SVI smile to (log_moneyness, total_variance) points. Returns a dict with
/// a, b, rho, m, sigma, rmse and max_abs_residual.
#[pyfunction]
fn svi_fit<'py>(
    py: Python<'py>,
    log_moneyness: PyReadonlyArrayDyn<'py, f64>,
    total_variance: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let ks: Vec<f64> = log_moneyness.as_array().iter().copied().collect();
    let ws: Vec<f64> = total_variance.as_array().iter().copied().collect();
    let fit = py.detach(|| qk::surface::fit_svi(&ks, &ws)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("a", fit.params.a)?;
    d.set_item("b", fit.params.b)?;
    d.set_item("rho", fit.params.rho)?;
    d.set_item("m", fit.params.m)?;
    d.set_item("sigma", fit.params.sigma)?;
    d.set_item("rmse", fit.rmse)?;
    d.set_item("max_abs_residual", fit.max_abs_residual)?;
    Ok(d)
}

/// Implied-volatility surface built from smiles, with interpolation and arbitrage checks.
#[pyclass(name = "VolSurface", module = "quantkit")]
struct PyVolSurface {
    inner: qk::surface::VolSurface,
}

fn report_dict<'py>(
    py: Python<'py>,
    r: &qk::surface::ArbitrageReport,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("calendar", r.calendar.len())?;
    d.set_item("butterfly", r.butterfly.len())?;
    d.set_item(
        "scope",
        match r.scope {
            qk::surface::ArbitrageScope::QuotedNodes => "quoted_nodes".to_string(),
            qk::surface::ArbitrageScope::SampledGrid {
                strikes_per_expiry,
                expiries,
            } => format!("sampled_grid({strikes_per_expiry} strikes x {expiries} expiries)"),
        },
    )?;
    Ok(d)
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

    /// Static-arbitrage check at the quoted nodes. Returns `{'calendar': n, 'butterfly': n,
    /// 'scope': 'quoted_nodes'}`. A clean result only covers the quoted nodes.
    fn check_arbitrage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        report_dict(py, &self.inner.check_arbitrage())
    }

    /// Static-arbitrage check on a sampled grid of the interpolated surface. A clean
    /// result covers only the sampled points.
    #[pyo3(signature = (strikes_per_expiry=50, extra_expiries_between=3))]
    fn check_arbitrage_sampled<'py>(
        &self,
        py: Python<'py>,
        strikes_per_expiry: usize,
        extra_expiries_between: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        report_dict(
            py,
            &self
                .inner
                .check_arbitrage_sampled(strikes_per_expiry, extra_expiries_between),
        )
    }
}

fn kind_mix(name: &str) -> PyResult<qk::dataset::KindMix> {
    match name {
        "both" => Ok(qk::dataset::KindMix::Both),
        "calls" => Ok(qk::dataset::KindMix::CallsOnly),
        "puts" => Ok(qk::dataset::KindMix::PutsOnly),
        _ => Err(PyValueError::new_err(
            "kinds must be 'both', 'calls' or 'puts'",
        )),
    }
}

/// Generates a reproducible synthetic option-pricing dataset (closed-form BSM labels).
/// Returns `{'columns': {name: ndarray}, 'metadata': {...}}`; see the Rust `dataset` module
/// for what the metadata and invariant report mean.
#[pyfunction]
#[pyo3(signature = (rows=10_000, seed=0, spot=100.0, moneyness=(0.5, 2.0), expiry=(1.0/52.0, 3.0),
                    volatility=(0.05, 1.0), rate=(-0.01, 0.06), dividend_yield=(0.0, 0.05),
                    kinds="both", latin_hypercube=true, include_boundary=true))]
#[allow(clippy::too_many_arguments)]
fn make_dataset<'py>(
    py: Python<'py>,
    rows: usize,
    seed: u64,
    spot: f64,
    moneyness: (f64, f64),
    expiry: (f64, f64),
    volatility: (f64, f64),
    rate: (f64, f64),
    dividend_yield: (f64, f64),
    kinds: &str,
    latin_hypercube: bool,
    include_boundary: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let spec = qk::dataset::DatasetSpec {
        rows,
        seed,
        spot,
        moneyness,
        expiry,
        volatility,
        rate,
        dividend_yield,
        kinds: kind_mix(kinds)?,
        sampling: if latin_hypercube {
            qk::dataset::Sampling::LatinHypercube
        } else {
            qk::dataset::Sampling::Uniform
        },
        include_boundary,
    };
    let ds = py.detach(|| qk::dataset::generate(&spec)).map_err(err)?;
    let cols = PyDict::new(py);
    type Pick = fn(&qk::dataset::DatasetRow) -> f64;
    let floats: [(&str, Pick); 12] = [
        ("spot", |r| r.spot),
        ("strike", |r| r.strike),
        ("rate", |r| r.rate),
        ("dividend_yield", |r| r.dividend_yield),
        ("volatility", |r| r.volatility),
        ("expiry", |r| r.expiry),
        ("price", |r| r.price),
        ("delta", |r| r.delta),
        ("gamma", |r| r.gamma),
        ("vega", |r| r.vega),
        ("theta", |r| r.theta),
        ("rho", |r| r.rho),
    ];
    for (name, f) in floats {
        let v: Vec<f64> = ds.rows.iter().map(f).collect();
        cols.set_item(name, v.into_pyarray(py))?;
    }
    let is_call: Vec<bool> = ds.rows.iter().map(|r| r.is_call).collect();
    let boundary: Vec<bool> = ds.rows.iter().map(|r| r.boundary).collect();
    cols.set_item("is_call", PyArray1::from_vec(py, is_call))?;
    cols.set_item("boundary", PyArray1::from_vec(py, boundary))?;

    let m = &ds.metadata;
    let inv = PyDict::new(py);
    inv.set_item("rows_checked", m.invariants.rows_checked)?;
    inv.set_item("non_finite", m.invariants.non_finite)?;
    inv.set_item("bound_violations", m.invariants.bound_violations)?;
    inv.set_item("parity_violations", m.invariants.parity_violations)?;
    inv.set_item("delta_violations", m.invariants.delta_violations)?;
    inv.set_item("sign_violations", m.invariants.sign_violations)?;
    inv.set_item("max_parity_error", m.invariants.max_parity_error)?;
    inv.set_item("all_passed", m.invariants.all_passed())?;
    let meta = PyDict::new(py);
    meta.set_item("quantkit_version", &m.quantkit_version)?;
    meta.set_item("label_model", &m.label_model)?;
    meta.set_item("conventions", &m.conventions)?;
    meta.set_item("sampler", &m.sampler)?;
    meta.set_item("sampled_rows", m.sampled_rows)?;
    meta.set_item("boundary_rows", m.boundary_rows)?;
    meta.set_item("checksum", m.checksum)?;
    meta.set_item("seed", seed)?;
    meta.set_item("invariants", inv)?;
    let out = PyDict::new(py);
    out.set_item("columns", cols)?;
    out.set_item("metadata", meta)?;
    Ok(out)
}

/// Sets the size of the global Rust thread pool used by batch functions. Must be called
/// before any parallel work; raises RuntimeError if the pool already exists. The
/// `RAYON_NUM_THREADS` environment variable does the same without code. Use 1 to avoid
/// oversubscribing CPUs when you already parallelise in Python.
#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<()> {
    if n == 0 {
        return Err(PyValueError::new_err("n must be at least 1"));
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .map_err(|e| PyRuntimeError::new_err(format!("thread pool already initialised: {e}")))
}

/// Human-readable name for a status code returned by the `*_detailed` functions.
#[pyfunction]
fn status_name(code: u8) -> PyResult<&'static str> {
    StatusCode::from_u8(code)
        .map(StatusCode::name)
        .ok_or_else(|| PyValueError::new_err("unknown status code"))
}

#[pymodule]
fn _quantkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("QuantError", py.get_type::<QuantError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    let status = PyDict::new(py);
    for code in [0u8, 1, 2, 3, 4, 5, 255] {
        if let Some(s) = StatusCode::from_u8(code) {
            status.set_item(s.name(), code)?;
        }
    }
    m.add("STATUS", status)?;
    m.add_class::<PyVolSurface>()?;
    m.add_function(wrap_pyfunction!(bs_price, m)?)?;
    m.add_function(wrap_pyfunction!(bs_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(implied_vol, m)?)?;
    m.add_function(wrap_pyfunction!(binomial_price, m)?)?;
    m.add_function(wrap_pyfunction!(mc_european, m)?)?;
    m.add_function(wrap_pyfunction!(mc_american, m)?)?;
    m.add_function(wrap_pyfunction!(mc_american_detailed, m)?)?;
    m.add_function(wrap_pyfunction!(bs_price_batch, m)?)?;
    m.add_function(wrap_pyfunction!(bs_price_detailed, m)?)?;
    m.add_function(wrap_pyfunction!(bs_greeks_batch, m)?)?;
    m.add_function(wrap_pyfunction!(implied_vol_batch, m)?)?;
    m.add_function(wrap_pyfunction!(implied_vol_detailed, m)?)?;
    m.add_function(wrap_pyfunction!(mc_european_detailed, m)?)?;
    m.add_function(wrap_pyfunction!(historical_var, m)?)?;
    m.add_function(wrap_pyfunction!(historical_cvar, m)?)?;
    m.add_function(wrap_pyfunction!(parametric_var, m)?)?;
    m.add_function(wrap_pyfunction!(sharpe_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(sortino_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(max_drawdown, m)?)?;
    m.add_function(wrap_pyfunction!(annualized_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(svi_fit, m)?)?;
    m.add_function(wrap_pyfunction!(make_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(status_name, m)?)?;
    Ok(())
}
