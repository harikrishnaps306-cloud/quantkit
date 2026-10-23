# quantkit

`quantkit` is a small Rust crate for options pricing and risk metrics. It prices European options with Black-Scholes-Merton (continuous dividend yield), American options on a Cox-Ross-Rubinstein tree, solves implied volatility, computes Greeks, and reports VaR, CVaR, Sharpe, Sortino and drawdown on a return series. Inputs are validated up front and every fallible call returns a `Result`, so bad data (negative volatility, zero expiry, NaN, empty slices) gives an error and never a panic. The crate is `#![forbid(unsafe_code)]` and has two dependencies, `thiserror` and `libm`.

## Install

```toml
[dependencies]
quantkit = "0.1"
```

Minimum supported Rust version: 1.74 (declared in `Cargo.toml`; only the current stable toolchain has been tested so far).

## Usage

Price a European option and its Greeks:

```rust
use quantkit::{bsm, greeks, OptionParams, OptionType};

let p = OptionParams::builder()
    .spot(42.0).strike(40.0).rate(0.10).volatility(0.20).expiry(0.5)
    .build()?;
let call = bsm::price(&p, OptionType::Call);          // ~4.76
let g = greeks::analytic(&p, OptionType::Call);       // delta, gamma, vega, theta, rho
```

Price an American put and recover an implied volatility:

```rust
use quantkit::{binomial, implied_vol, ExerciseStyle, OptionParams, OptionType};

let p = OptionParams::builder()
    .spot(50.0).strike(50.0).rate(0.10).volatility(0.40).expiry(5.0 / 12.0)
    .build()?;
let american = binomial::price(&p, OptionType::Put, ExerciseStyle::American, 500)?;

let contract = OptionParams::builder()
    .spot(21.0).strike(20.0).rate(0.10).expiry(0.25)
    .build_contract()?;
let iv = implied_vol::solve(&contract, OptionType::Call, 1.875)?;   // ~0.235
```

Risk metrics on a slice of returns:

```rust
use quantkit::risk;

let returns = [0.004, -0.011, 0.007, 0.001, -0.023, 0.015, -0.005, 0.009];
let var = risk::historical_var(&returns, 0.95)?;     // positive loss fraction
let cvar = risk::historical_cvar(&returns, 0.95)?;
let sharpe = risk::sharpe_ratio(&returns, 0.0, 252.0)?;
```

Runnable versions live in [`examples/`](examples): `cargo run --example price_with_greeks` and `cargo run --example var_from_returns`.

## Features

- **Pricing:** BSM European call/put with continuous dividend yield; CRR binomial tree for American and European options.
- **Implied volatility:** Newton-Raphson with bisection fallback; explicit `PriceOutOfBounds` and `NoConvergence` errors.
- **Greeks:** analytic BSM delta, gamma, vega, theta, rho; generic bump-and-reprice Greeks (used for the binomial pricer).
- **Risk:** historical VaR and CVaR, parametric (normal) VaR, Sharpe, Sortino, max drawdown, annualised volatility.
- **Design:** validated builder types, no panics on bad input, borrowed slices for return series, doc example on every public item.

Conventions worth knowing: vega is per 1.00 of volatility, rho per 1.00 of rate, theta per year. VaR and CVaR are positive loss fractions. See the module docs for details.

## Accuracy and validation

- BSM prices are checked against Hull's textbook example (S=42, K=40, r=10%, sigma=20%, T=0.5: call 4.76, put 0.81), Hull's Greeks example (S=49, K=50, r=5%, sigma=20%, T=20 weeks) and Hull's implied-volatility example (call 1.875 gives about 23.5%). Hull values are published to two or three digits, so those tests use matching tolerances.
- Higher-precision checks (to about 1e-11) compare against an independent Python implementation built on `math.erfc`. These guard against implementation slips; they are not independent derivations of the formulas.
- The binomial pricer reproduces Hull's two-step American put (7.43) and five-step American put (4.49), and converges to BSM as steps increase (CRR oscillates, so convergence is not monotone).
- Property tests (proptest) cover put-call parity, monotonicity in spot, delta bounds, American >= European, no-arbitrage price bounds, implied-vol round trips, and VaR/CVaR/drawdown invariants.
- The normal CDF uses `libm::erfc` (about 1 ulp, with relative accuracy in the tails).

Limits: the CRR tree is `O(n^2)` and its Greeks are finite differences, so they are noisier than the analytic ones. Implied volatility is ill-conditioned for deep in-the-money options priced at intrinsic value; the solver returns a volatility that reprices the option, which need not equal the volatility you started from.

This crate is provided for education and research, with no warranty, and is not investment advice.

## Roadmap

Not in v1: Monte Carlo pricing, exotic options, stochastic volatility models, volatility surfaces, Python bindings, market data feeds.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
