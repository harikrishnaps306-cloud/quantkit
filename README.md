# quantkit

`quantkit` is an unpublished Rust crate (with Python/NumPy bindings) for options pricing and investment risk. It prices European options with Black-Scholes-Merton (continuous dividend yield), American options on a Cox-Ross-Rubinstein tree and with Longstaff-Schwartz Monte Carlo, solves implied volatility, computes Greeks, builds and checks volatility surfaces, and reports VaR, CVaR, Sharpe, Sortino and drawdown. Two things are meant to set it apart: **batch results that say why each row failed** (status code plus solver or Monte Carlo diagnostics, instead of a bare `NaN`) and **reproducible, self-auditing synthetic datasets** for pricing models. Inputs are validated against a documented supported domain, every fallible call returns a `Result`, and the core is `#![forbid(unsafe_code)]` with two required dependencies (`thiserror`, `libm`).

Status: version 0.2.0, **not published** to crates.io or PyPI, repository URL not set. See [docs/NUMERICS.md](docs/NUMERICS.md) for exact conventions and guarantees, [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for measured performance, and [docs/market-gap-report.md](docs/market-gap-report.md) for how the feature choices were made.

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
// r.price / r.std_error: independent paths (low-biased). r.in_sample_price: overfit, high-biased.

let ds = dataset::generate(&dataset::DatasetSpec { rows: 10_000, seed: 7, ..Default::default() })?;
assert!(ds.metadata.invariants.all_passed());   // ds.metadata.checksum, spec, conventions, version
```

Runnable examples are in [`examples/`](examples): `price_with_greeks`, `var_from_returns`, `monte_carlo_and_surface`, `ml_dataset` (run with `cargo run --example <name>`).

### Python

```bash
cd python && pip install maturin numpy && maturin develop --release
```

```python
import numpy as np, quantkit as qk

qk.bs_price(42, 40, 0.10, 0.20, 0.5)                                    # 4.759... (float)
prices = qk.bs_price_batch(np.linspace(80, 120, 1000), 100.0, 0.03, 0.2, 1.0)   # ndarray, any shape, GIL released
d = qk.implied_vol_detailed(prices, spots, 100.0, 0.03, 1.0)            # d["vol"], d["status"], d["residual"], d["iterations"], d["vega"], d["messages"]
qk.STATUS                                                                # {'Ok': 0, 'InvalidInput': 1, ...}
ds = qk.make_dataset(rows=10_000, seed=7)                                # {'columns': {...}, 'metadata': {...}}
```

Array arguments must share a shape, scalars broadcast, and all-scalar calls return scalars. `errors="raise"` (default) raises `quantkit.QuantError` naming the first bad row; `errors="nan"` returns `NaN` for failed rows. Set `RAYON_NUM_THREADS` (or call `quantkit.set_num_threads(n)` first) if you also parallelise in Python. The wheel is built with the stable ABI for Python 3.9 and later; only Python 3.11 on Windows has been tested locally.

## What it contains

* **Pricing:** BSM European; CRR tree (American/European, Bermudan with `steps` dates); seeded Monte Carlo European; Longstaff-Schwartz with independent training and evaluation paths.
* **Greeks:** analytic BSM; central-difference numerical Greeks for any pricer.
* **Implied volatility:** Newton with bisection fallback; precisely stated guarantee; diagnostics (residual, steps, vega).
* **Surfaces:** build from vols or prices; interpolation; scoped arbitrage checks (quoted nodes, or a sampled grid); raw SVI with the Gatheral-Jacquier butterfly condition; SVI fitting with residual diagnostics.
* **Risk:** historical VaR/CVaR, parametric VaR, Sharpe, Sortino, max drawdown, annualised volatility, with documented definitions.
* **Batch:** plain batch functions; `detailed` per-row results with status codes; column-oriented fast paths; optional rayon parallelism; option grids.
* **Datasets:** seeded Latin-hypercube or uniform sampling, boundary-corner rows, closed-form labels, metadata (spec, version, conventions, checksum) and a financial-invariant report.
* **Optional:** `serde` (inputs are re-validated on deserialize), `parallel`.

## Verification status

Executed locally (Windows 11, Rust 1.97.1; Python 3.11.9):

* `cargo fmt --check`, `cargo clippy --all-targets --all-features -D warnings`, `cargo doc` with warnings denied, all tests with default features, `--features serde`, `--features parallel` and all features, including 98 unit tests, property tests, a 30,000-case-per-test no-panic suite, domain-boundary tests, independent-reference tests and doctests.
* Library unit tests on Rust 1.74 with all features.
* Wheel built and installed in a **clean virtualenv**; 19 Python tests pass against the installed wheel.
* Benchmarks in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

Configured but **not yet executed**: the GitHub Actions workflow (Linux/macOS/Windows, Python 3.9/3.11/3.13, MSRV job). No claim is made that it passes.

Reference values: BSM against an independent `erf` closed form for the ATM zero-rate case, the exact call-put symmetry `C(S,K,r,q) = P(K,S,q,r)`, homogeneity, and 5-point finite differences of prices for every Greek; the binomial tree against exact `r=0`/`q=0` early-exercise identities and McKean's perpetual put; the SVI butterfly density against second derivatives of Black prices. Hull's textbook figures are used only where noted in the tests, with tolerances matching their published precision (Hull's two- and five-step American-put figures and rho could not be confirmed against the book and are **not** used as references).

## Known limitations

* Supported input domain is bounded ([docs/NUMERICS.md](docs/NUMERICS.md) section 1); zero volatility and zero expiry are rejected rather than approximated.
* Implied-volatility accuracy is conditioned by vega: the solver guarantees the *price residual*, not the recovered volatility. Very deep out-of-the-money prices (about `1e-300`) fail with `NoConvergence`. The solver is Newton/bisection, not Jaeckel's "Let's Be Rational".
* Monte Carlo `std_error` is sampling error only. Plain Monte Carlo is unreliable for deep out-of-the-money payoffs. Longstaff-Schwartz is a Bermudan, low-biased estimate; policy error is not measured. Random streams are not provably disjoint (hash-seeded, no jump-ahead), and the generator is tested statistically, not against reference vectors.
* Surface arbitrage reports cover only the stated scope; a clean report is never a global guarantee. No SSVI. SVI fits do not enforce the butterfly condition.
* Datasets are closed-form European Black-Scholes labels: no smile, no noise, no American or exotic labels. The invariant report is a self-consistency check, not an independent validation.
* Bitwise cross-platform reproducibility covers `mc`, `dist`, `bsm`, `greeks` and `dataset` (pure `libm`); it has been checked on one OS only. The tree, solver, risk and surface modules can differ in the last place across platforms.
* Python: GIL-released batch functions share a global Rayon pool; single-thread throughput is comparable to a NumPy/SciPy implementation, not better ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)).

## Roadmap

Next candidates (see the final report in the repository history for reasoning): a faster, more tail-robust implied-volatility method; a published CI run on all three operating systems and a first PyPI/crates.io release once a repository exists; jump-ahead random streams; parallel dataset generation; American and smile-aware dataset labels. Not planned for now: live data, trading, an LLM tool server, bonds, portfolio optimisation, SSVI.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option. Provided for education and research, with no warranty; not investment advice.
