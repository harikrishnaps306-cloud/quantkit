# quantkit

[![CI](https://github.com/harikrishnaps306-cloud/quantkit/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/harikrishnaps306-cloud/quantkit/actions/workflows/ci.yml)

`quantkit` is a Rust crate (with Python/NumPy bindings, published on PyPI as `quantkit-rs`) for options pricing and investment risk. It prices European options with Black-Scholes-Merton (continuous dividend yield), American options on a Cox-Ross-Rubinstein tree and with Longstaff-Schwartz Monte Carlo, solves implied volatility, computes Greeks, builds and checks volatility surfaces, and reports VaR, CVaR, Sharpe, Sortino and drawdown. Two things are meant to set it apart: **batch results that say why each row failed** (status code plus solver or Monte Carlo diagnostics, instead of a bare `NaN`) and **reproducible, self-auditing synthetic datasets** for pricing models. Inputs are validated against a documented supported domain, every fallible call returns a `Result`, and the core is `#![forbid(unsafe_code)]` with two required dependencies (`thiserror`, `libm`).

Status: **initial experimental release candidate, version 0.2.0**. Version 0.2.0 is on PyPI as `quantkit-rs` (`pip install quantkit-rs`) and is not yet on crates.io. The GitHub Actions workflow has run green once on Linux, macOS and Windows (details and limits in [docs/compatibility.md](docs/compatibility.md)). The public API is `0.x`: breaking changes may occur in any minor release and only there; see [docs/VERSIONING.md](docs/VERSIONING.md) for the policy, [docs/RELEASING.md](docs/RELEASING.md) for how releases are made and [docs/RELEASE_NOTES_0.2.0.md](docs/RELEASE_NOTES_0.2.0.md) for what is most likely to move. See [docs/NUMERICS.md](docs/NUMERICS.md) for exact conventions and guarantees, [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for measured performance, [docs/market-gap-report.md](docs/market-gap-report.md) for how the feature choices were made, and [docs/release-readiness.md](docs/release-readiness.md) for what has and has not been validated.

## Install

Not on crates.io yet. From a checkout:

```toml
[dependencies]
quantkit = { path = "../quantkit" }
# optional: features = ["serde", "parallel"]
```

Minimum supported Rust version: 1.74. The library with all features builds and passes its unit tests on Rust 1.74 when dependencies are resolved with a Rust-version-aware lockfile (`CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback`); without that, a fresh resolve may pick releases that need a newer compiler. Integration tests, benchmarks and the Python bindings are only exercised on current stable.

## Usage

```rust
use quantkit::{bsm, greeks, OptionParams, OptionType};

let p = OptionParams::builder()
    .spot(42.0).strike(40.0).rate(0.10).volatility(0.20).expiry(0.5)
    .build()?;                                        // validated against quantkit::limits
let call = bsm::price(&p, OptionType::Call);          // ~4.76
let g = greeks::analytic(&p, OptionType::Call);       // delta, gamma, vega (per 1.00), theta (per year), rho (per 1.00)
```

Implied volatility with diagnostics, and American pricing:

```rust
use quantkit::{binomial, implied_vol, ExerciseStyle, OptionParams, OptionType};

let contract = OptionParams::builder()
    .spot(21.0).strike(20.0).rate(0.10).expiry(0.25)
    .build_contract()?;
let sol = implied_vol::solve_detailed(&contract, OptionType::Call, 1.875, Default::default())?;
// sol.vol ~ 0.235, sol.residual, sol.iterations, sol.newton_steps, sol.vega

let p = OptionParams::builder()
    .spot(50.0).strike(50.0).rate(0.10).volatility(0.40).expiry(5.0 / 12.0)
    .build()?;
let put = binomial::price(&p, OptionType::Put, ExerciseStyle::American, 500)?;
```

Per-row batch results (a failed row never poisons the others):

```rust
use quantkit::detailed::{self, RawQuote};
use quantkit::{implied_vol::SolverConfig, OptionType, StatusCode};

let quotes = [
    RawQuote { spot: 21.0, strike: 20.0, rate: 0.1, dividend_yield: 0.0, expiry: 0.25, price: 1.875 },
    RawQuote { spot: 21.0, strike: 20.0, rate: 0.1, dividend_yield: 0.0, expiry: 0.25, price: 25.0 }, // impossible
];
let rows = detailed::implied_vol(&quotes, OptionType::Call, SolverConfig::default());
assert_eq!(rows[0].status, StatusCode::Ok);                 // rows[0].residual, .iterations, .vega
assert_eq!(rows[1].status, StatusCode::PriceOutOfBounds);   // rows[1].error holds the details
```

Monte Carlo with an independently evaluated exercise policy, and a reproducible dataset:

```rust
use quantkit::{dataset, mc, OptionParams, OptionType};

let p = OptionParams::builder().spot(36.0).strike(40.0).rate(0.06).volatility(0.2).expiry(1.0).build()?;
let train = mc::McConfig { paths: 20_000, seed: 1, antithetic: true };
let eval  = mc::McConfig { paths: 100_000, seed: 2, antithetic: true };
let r = mc::american_lsm_detailed(&p, OptionType::Put, 50, train, eval)?;
// r.price / r.std_error: value of the trained policy on independent paths, and the sampling error
// of that mean given the policy. r.in_sample_price: measured on the training paths (optimistic).

let ds = dataset::generate(&dataset::DatasetSpec { rows: 10_000, seed: 7, ..Default::default() })?;
assert!(ds.metadata.invariants.all_passed());   // ds.metadata.checksum, spec, conventions, version
```

Runnable examples are in [`examples/`](examples): `price_with_greeks`, `var_from_returns`, `monte_carlo_and_surface`, `ml_dataset` (run with `cargo run --example <name>`).

### Python

```bash
cd python && pip install maturin numpy && maturin develop --release
```

The PyPI distribution is named `quantkit-rs` (the name `quantkit` belongs to an unrelated project); the import name is `quantkit`: `pip install quantkit-rs`, then `import quantkit`.

```python
import numpy as np, quantkit as qk

qk.bs_price(42, 40, 0.10, 0.20, 0.5)                                    # 4.759... (float)
prices = qk.bs_price_batch(np.linspace(80, 120, 1000), 100.0, 0.03, 0.2, 1.0)   # ndarray, any shape, GIL released
d = qk.implied_vol_detailed(prices, spots, 100.0, 0.03, 1.0)            # d["vol"], d["status"], d["residual"], d["iterations"], d["vega"], d["messages"]
qk.STATUS                                                                # {'Ok': 0, 'InvalidInput': 1, ...}
ds = qk.make_dataset(rows=10_000, seed=7)                                # {'columns': {...}, 'metadata': {...}}
```

Array arguments must share a shape, scalars broadcast, and all-scalar calls return scalars. `errors="raise"` (default) raises `quantkit.QuantError` naming the first bad row; `errors="nan"` returns `NaN` for failed rows. Set `RAYON_NUM_THREADS` (or call `quantkit.set_num_threads(n)` first) if you also parallelise in Python. The wheel is built once with the Python 3.9 stable ABI and declares `Requires-Python >=3.9` and `numpy>=1.21.6` (1.21.6 is the oldest NumPy release actually tested; earlier 1.21 releases are untested, not known-bad). The exact combinations that were executed (CPython 3.9.25, 3.10.20, 3.11.9, 3.12.13, 3.13.14 with NumPy 1.21.6, 1.23.5, 1.26.4, 2.0.2, 2.4.6, 2.5.4; ten pairs, Windows x86-64 only), the CI pairs that are configured but have not run, and what is untested are listed in [docs/compatibility.md](docs/compatibility.md).

## What it contains

* **Pricing:** BSM European; CRR tree (American/European, Bermudan with `steps` dates); seeded Monte Carlo European; Longstaff-Schwartz with independent training and evaluation paths.
* **Greeks:** analytic BSM; central-difference numerical Greeks for any pricer.
* **Implied volatility:** log-domain Newton on the normalised Black function with a bisection fallback, accurate into the far tails; precisely stated guarantee; diagnostics (residual, steps, vega).
* **Surfaces:** build from vols or prices; interpolation; scoped arbitrage checks (quoted nodes, or a sampled grid); raw SVI with the Gatheral-Jacquier butterfly condition; SVI fitting with residual diagnostics.
* **Risk:** historical VaR/CVaR, parametric VaR, Sharpe, Sortino, max drawdown, annualised volatility, with documented definitions.
* **Batch:** plain batch functions; `detailed` per-row results with status codes; column-oriented fast paths; optional rayon parallelism; option grids.
* **Datasets:** seeded Latin-hypercube or uniform sampling, boundary-corner rows, closed-form labels, metadata (spec, version, conventions, checksum) and a financial-invariant report.
* **Optional:** `serde` (inputs are re-validated on deserialize), `parallel`.

## Verification status

Everything below was executed on one machine (Windows 11, x86-64). Commands, toolchain and dependency versions, and test counts are recorded in [docs/release-readiness.md](docs/release-readiness.md).

* **Rust 1.97.1:** `cargo fmt --check`, clippy with warnings denied (library and bindings), `cargo doc` with warnings denied, tests with default features (158), `--features serde` (159), `--features parallel` (158) and all features (159), 93 doctests, the README snippets and all four examples, `cargo package` with the packaged source built.
* **Rust 1.74.1 (the declared minimum), with a Rust-version-aware lockfile:** the library, all tests, benches and examples compile, and 123 unit tests, 36 integration tests and 93 doctests pass. How to reproduce: [docs/release-readiness.md](docs/release-readiness.md).
* **Python:** one wheel in clean virtualenvs on ten Python/NumPy combinations (20 tests each, all pass; listed in [docs/compatibility.md](docs/compatibility.md)) and a source distribution built and installed in a clean virtualenv (20 tests pass).
* **Evidence quality:** 150-digit reference tests for the pricer and solver, independent-relation tests, a no-panic suite (run once at 30,000 cases per test; 400 by default), and repeated property tests. These are finite samples, not proofs.

**Not executed:** the GitHub Actions workflow (Linux, macOS, Windows jobs, the MSRV job, the Python matrix, the sdist job). It is configured to run on pushes to `main` or `master`, on pull requests, and manually; until it has run, nothing here is validated on Linux or macOS. Also untested: ARM, other `libm` versions, and any real-world use.

Reference values: the normalised Black function and implied volatilities against 150-digit arithmetic over `|d1|` up to 36 (`tests/black_accuracy.rs`); BSM against an independent `erf` closed form for the ATM zero-rate case, the exact call-put symmetry `C(S,K,r,q) = P(K,S,q,r)`, homogeneity, and 5-point finite differences of prices for every Greek; the binomial tree against exact early-exercise identities under their stated sufficient conditions and McKean's perpetual put; the SVI butterfly density against second derivatives of Black prices. Hull's textbook figures are used only where noted in the tests, with tolerances matching their published precision (Hull's two- and five-step American-put figures and rho could not be confirmed against the book and are **not** used as references).

## Known limitations

* Supported input domain is bounded ([docs/NUMERICS.md](docs/NUMERICS.md) section 1); zero volatility and zero expiry are rejected rather than approximated.
* Implied-volatility accuracy is conditioned by vega: the solver guarantees the *price residual*; out-of-the-money volatilities are recovered to about `1e-13` (checked against 150-digit prices down to `1e-290`), but in-the-money options priced almost at intrinsic value are inherently ill-conditioned. The solver uses the normalised-Black, log-domain formulation of "Let's Be Rational" with a simpler iteration (several evaluations, not two) and is not that algorithm.
* Monte Carlo `std_error` is sampling error only. Plain Monte Carlo is unreliable for deep out-of-the-money payoffs. Longstaff-Schwartz prices a Bermudan option with a learned policy: in expectation its price does not exceed the Bermudan value, `std_error` is conditional on the trained policy, and policy, discretisation and model error are not included. Random streams have distinct hash-derived initial states but are not proved non-overlapping (no jump-ahead). The generator reproduces the published `xoshiro256++` reference vector; the normal/uniform checks are statistical.
* Surface arbitrage reports cover only the stated scope; a clean report is never a global guarantee. No SSVI. SVI fits do not enforce the butterfly condition.
* Datasets are closed-form European Black-Scholes labels: no smile, no noise, no American or exotic labels. The invariant report is a self-consistency check, not an independent validation.
* Reproducibility: `mc`, `dist`, `bsm`, `black`, `greeks` and `dataset` are written to avoid platform dependence (pure `libm`), and identical bits were observed on one target (`x86_64-pc-windows-msvc`) with Rust 1.97.1 and 1.74.1 and `libm` 0.2.16. Other targets and later dependency versions are not checked; the CI that would check Linux and macOS has not run. The tree, solver bounds, risk and surface modules can differ in the last place across platforms.
* Python: GIL-released batch functions share a global Rayon pool; single-thread throughput of the accurate pricer is about half that of a NumPy/SciPy implementation on wide-ranging inputs; with all threads it is about 2.6x faster ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)). Accuracy, not speed, is the design priority of the pricer and the implied-volatility solver.

## Roadmap

Next candidates (see the final report in the repository history for reasoning): Jaeckel's rational starting guess and higher-order step to cut implied-volatility evaluations; a published CI run on all three operating systems and a first PyPI/crates.io release once a repository exists; jump-ahead random streams; parallel dataset generation; American and smile-aware dataset labels. Not planned for now: live data, trading, an LLM tool server, bonds, portfolio optimisation, SSVI.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option. Provided for education and research, with no warranty; not investment advice.
