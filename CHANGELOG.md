# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0]

Unreleased locally; nothing here has been published, so the "Changed" items are
differences from the earlier 0.2.0 development snapshot (commit `02a6393`), listed
so that code written against it can be migrated.

### Added
- `limits`: the supported input domain, enforced by `OptionParams`, `Contract`, `VolSurface::new` and `dataset`.
- `StatusCode` and `QuantError::status_code()`; `detailed` module with per-row `PriceRow`, `GreeksRow`, `IvRow`, `McRow` and column-oriented `price_columns` / `greeks_columns`.
- `implied_vol::solve_detailed` returning `IvSolution` (residual, Newton/bisection step counts, vega).
- `mc::american_lsm_detailed` / `LsmResult`: independent training and evaluation paths, in-sample versus out-of-sample price, fit diagnostics. `mc::Rng::for_stream`, `mc::derive_seed`.
- `dataset` module: seeded, audited synthetic option datasets (`generate`, `DatasetSpec`, `Dataset`, `InvariantReport`, CSV export, checksum).
- `surface`: `check_arbitrage_sampled`, `ArbitrageScope`, vertical-spread (slope) condition, `SviFit::max_abs_residual`.
- Python: n-dimensional arrays with scalar pass-through, `*_detailed` functions, `STATUS`, `status_name`, `make_dataset`, `mc_american_detailed`, `set_num_threads`.
- `docs/NUMERICS.md`, `docs/BENCHMARKS.md`, `docs/market-gap-report.md`; throughput benchmarks; independent-reference, domain-boundary and no-panic test suites.

### Changed (breaking relative to the earlier snapshot)
- Inputs outside `quantkit::limits` are rejected (for example volatility below `1e-8`, spot above `1e12`, `|rate| * expiry > 500`).
- `bsm::price` clamps rounding-sized negative results to zero.
- Implied volatility: attainable-range failures are `PriceOutOfBounds` (previously some were `NoConvergence`); the documentation no longer claims unconditional convergence.
- Monte Carlo: results changed (block-based streams, `libm` functions); `std_error` for antithetic Longstaff-Schwartz is computed over pair means; `american_lsm` now returns the **out-of-sample** estimate trained on independent paths. European Monte Carlo and dataset results are thread-count independent and bitwise reproducible across platforms.
- `ArbitrageReport` gained a `scope` field; its documentation now states that clean reports are scoped.
- Risk: scale-safe standard deviations; results that would overflow are errors; max drawdown rejects equity paths that overflow.
- `greeks::numerical` rejects non-positive or non-finite bump sizes.
- Python: all-scalar batch calls return scalars (previously one-element arrays); array arguments must share a shape; `check_arbitrage()` returns an extra `scope` entry.
- Removed the guessed `repository` URL from `Cargo.toml`; set the real one before publishing.

### Fixed
- Integer-overflow panics in binomial and Monte Carlo size handling (found by the no-panic suite).
- A panic in surface arbitrage checks for nodes outside the option domain.
- Valid-but-extreme inputs that produced `NaN`/negative closed-form prices (6,157 of 20,480 probed combinations produced non-finite values before the domain was defined).
- Spurious zero-variance errors for tiny-scale returns, and silent `inf`/wrong drawdown for huge returns.
- In-sample optimism of Longstaff-Schwartz estimates.

## [0.1.0]

First release.

### Added
- `OptionParams` / `Contract` validated option inputs with a builder.
- `bsm`: Black-Scholes-Merton European call and put prices with a continuous dividend yield.
- `greeks`: analytic BSM delta, gamma, vega, theta, rho; generic bump-and-reprice numerical Greeks.
- `implied_vol`: Newton-Raphson solver with bisection fallback and explicit errors.
- `binomial`: Cox-Ross-Rubinstein tree for American and European options, with numerical Greeks.
- `risk`: historical VaR/CVaR, parametric VaR, Sharpe, Sortino, max drawdown, annualised volatility.
- `dist`: standard normal pdf, cdf and inverse cdf.
- Property tests (proptest), criterion benchmarks, two runnable examples, CI workflow.
