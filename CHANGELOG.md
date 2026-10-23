# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
