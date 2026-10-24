# quantkit 0.2.0: initial experimental release (PyPI `quantkit-rs`; not yet on crates.io)

First public release, experimental. Published to PyPI as `quantkit-rs` 0.2.0 on
2026-10-24; not published to crates.io. Tag `v0.2.0` is commit `df7d034`.

## What this is

`quantkit` is a Rust library, with optional Python/NumPy bindings, for options
pricing and basic investment risk:

* Black-Scholes-Merton European prices and analytic Greeks, with a numerically
  stable evaluation far out of the money.
* Cox-Ross-Rubinstein binomial pricing (American/European), and seeded Monte
  Carlo (European, and Longstaff-Schwartz American with the exercise policy
  trained on one set of paths and evaluated on independent paths).
* Implied volatility with diagnostics (residual, iterations, vega, and the
  elasticity that says how well the price determines the volatility).
* Volatility surfaces with explicitly scoped arbitrage checks, and SVI fitting.
* Historical and parametric VaR/CVaR, Sharpe, Sortino, drawdown, volatility.
* Batch APIs whose rows carry a status code and diagnostics instead of a bare
  `NaN`, optional rayon parallelism, optional `serde`, and a seeded, self-auditing
  synthetic-dataset generator.

## Please read before using

* **Experimental.** Version `0.2.0`, never used outside its own tests. Do not
  rely on it for trading, risk limits, or any decision with financial
  consequences. It is provided for education and research, with no warranty.
* **API stability.** `0.x` semantic versioning: breaking changes may occur in
  any minor release and only there; patch releases fix bugs and add things
  without breaking documented behaviour. The full policy, including what counts
  as breaking and which parts are most likely to move, is in
  `docs/VERSIONING.md`.
* **Numerical outputs can change between versions.** Regression-locked values
  (the Monte Carlo and dataset golden vectors) were deliberately re-recorded
  during development when the random-number mapping and the pricer changed, and
  may be again. Do not assume results stay bitwise identical across releases,
  compilers or dependency upgrades.
* **Supported input domain is bounded** (`quantkit::limits`). Out-of-range
  inputs return an error; zero volatility and zero expiry are rejected.
* **Accuracy is scoped.** The pricer and implied-volatility solver are checked
  against 150-digit arithmetic on finite sets of inputs; that is evidence, not a
  proof. The accurate pricer is slower than a naive one (see
  `docs/BENCHMARKS.md`). Monte Carlo standard errors are sampling errors only.
  Surface arbitrage checks cover the quoted points or a sampled grid, never the
  whole surface; SVI fits do not enforce the butterfly condition.

## Compatibility actually tested

| item | result |
|---|---|
| Rust 1.97.1 (Windows) | all checks pass |
| Rust 1.74.1 (declared MSRV, Windows), with a Rust-version-aware lockfile | library, tests, benches, examples and doctests pass |
| CPython 3.9-3.13 with NumPy 1.22.0 to 2.5.4 (Windows x86-64): exact pairs listed in `docs/compatibility.md` | all pass; declared `numpy>=1.22.0` (the oldest tested; earlier releases carry a known advisory) |
| Linux (x86-64), macOS (arm64) | CI run on GitHub, all jobs green at commit `9a28f6e`; golden vectors matched. Only what the workflow executes was tested. |
| ARM Linux, musl | **not tested** |

Details and exact versions: `docs/release-readiness.md`.

## Known limitations

* CI has run once to green (`9a28f6e`); only the combinations in `docs/compatibility.md` are covered, and the published wheels were installed from PyPI on GitHub-hosted Linux, macOS and Windows runners only (six rows in `docs/compatibility.md`), not on other machines.
* Bitwise reproducibility of Monte Carlo and dataset outputs was observed on one
  target only; the design (pure `libm`, fixed evaluation order) is not a proof.
* The solver is not Jaeckel's "Let's Be Rational" and needs several evaluations.
* Hull's two- and five-step American-put figures and rho could not be confirmed
  against the book and are not used as references.
* No SSVI, no exotics, no stochastic-volatility models, no live data.

## How to try it

```bash
# Rust: from a checkout
cargo add --path path/to/quantkit          # or a path dependency in Cargo.toml
cargo run --example price_with_greeks

# Python: from PyPI (Linux x86-64, macOS arm64, Windows x86-64 wheels; others build the sdist)
pip install quantkit-rs

# Python: or build and install a wheel from a checkout
cd python && pip install maturin numpy && maturin develop --release
```

## Upgrade notes

None: this is the first release. The changelog lists the changes made during
development relative to earlier internal snapshots.
