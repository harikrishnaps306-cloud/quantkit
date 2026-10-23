# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0]

### Added
- `mc`: seeded xoshiro256++ RNG, European Monte Carlo (antithetic variates, standard error, confidence interval) and Longstaff-Schwartz American pricing.
- `surface`: `VolSurface` (from vols or prices), interpolation, calendar/butterfly arbitrage checks, raw SVI parameters with the Gatheral-Jacquier butterfly condition, and quasi-explicit SVI fitting.
- `batch`: batch price, Greeks, binomial and implied-vol functions; `grid` for option-grid generation.
- Cargo features `serde` (validated deserialization of option inputs) and `parallel` (rayon-backed batch functions).
- `python/`: PyO3 + maturin bindings with NumPy batch functions (not yet published).

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
