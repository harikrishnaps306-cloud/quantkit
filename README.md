# quantkit

`quantkit` is a Rust crate for options pricing and risk metrics. It prices European options with Black-Scholes-Merton (continuous dividend yield), American options on a Cox-Ross-Rubinstein tree, solves implied volatility, computes Greeks, and reports VaR, CVaR, Sharpe, Sortino and drawdown on a return series. It also includes seeded Monte Carlo (European and Longstaff-Schwartz American), batch and parallel APIs, implied-volatility surfaces with arbitrage checks and SVI fitting, optional `serde` support, and Python bindings. Inputs are validated up front and every fallible call returns a `Result`, so bad data (negative volatility, zero expiry, NaN, empty slices) gives an error and never a panic. The core crate is `#![forbid(unsafe_code)]` and has two required dependencies, `thiserror` and `libm`.

## Install

```toml
[dependencies]
quantkit = "0.2"

# optional features
# quantkit = { version = "0.2", features = ["serde", "parallel"] }
```

Minimum supported Rust version: 1.74 (declared in `Cargo.toml`; only the current stable toolchain has been tested so far, and the optional `serde` and `rayon` dependencies may need something newer).

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

Monte Carlo and a vol surface:

```rust
use quantkit::{mc, surface::{SurfaceSlice, VolSurface}, OptionParams, OptionType};

let p = OptionParams::builder().spot(100.0).strike(105.0).rate(0.03).volatility(0.25).expiry(1.0).build()?;
let est = mc::european(&p, OptionType::Call, mc::McConfig::default())?;   // price + std_error, seeded
let american_put = mc::american_lsm(&p, OptionType::Put, 50, mc::McConfig::default())?;

let surface = VolSurface::new(100.0, 0.02, 0.0, vec![
    SurfaceSlice { expiry: 0.25, strikes: vec![90.0, 100.0, 110.0], vols: vec![0.26, 0.21, 0.23] },
    SurfaceSlice { expiry: 1.00, strikes: vec![80.0, 100.0, 120.0], vols: vec![0.27, 0.22, 0.24] },
])?;
let v = surface.vol(100.0, 0.5)?;
assert!(surface.check_arbitrage().is_clean());
```

Runnable versions live in [`examples/`](examples): `cargo run --example price_with_greeks`, `var_from_returns` and `monte_carlo_and_surface`.

### Python

```bash
cd python && pip install maturin numpy && maturin develop --release
```

```python
import numpy as np, quantkit as qk

qk.bs_price(42, 40, 0.10, 0.20, 0.5)                                    # 4.759...
qk.bs_price_batch(np.linspace(80, 120, 1000), 100.0, 0.03, 0.2, 1.0)    # NumPy in, NumPy out, GIL released
qk.implied_vol_batch(prices, spots, 100.0, 0.03, 1.0, errors="nan")     # NaN for unsolvable rows
```

Errors raise `quantkit.QuantError` (a `ValueError`). The bindings are not yet on PyPI.

## Features

- **Pricing:** BSM European call/put with continuous dividend yield; CRR binomial tree for American and European options.
- **Implied volatility:** Newton-Raphson with bisection fallback; explicit `PriceOutOfBounds` and `NoConvergence` errors.
- **Greeks:** analytic BSM delta, gamma, vega, theta, rho; generic bump-and-reprice Greeks (used for the binomial pricer).
- **Monte Carlo:** seeded GBM European pricing with antithetic variates, standard errors and confidence intervals; Longstaff-Schwartz American pricing.
- **Vol surfaces:** build from prices or vols, interpolate (linear in total variance), detect calendar and butterfly arbitrage at the nodes, fit SVI smiles.
- **Batch and ML-friendly:** `batch` module (optional `parallel` feature via rayon), option-grid generation, optional `serde` (inputs are re-validated on deserialize), NumPy-based Python bindings.
- **Risk:** historical VaR and CVaR, parametric (normal) VaR, Sharpe, Sortino, max drawdown, annualised volatility.
- **Design:** validated builder types, no panics on bad input, borrowed slices for return series, doc example on every public item.

Conventions worth knowing: vega is per 1.00 of volatility, rho per 1.00 of rate, theta per year. VaR and CVaR are positive loss fractions. See the module docs for details.

## Accuracy and validation

- BSM prices are checked against Hull's textbook example (S=42, K=40, r=10%, sigma=20%, T=0.5: call 4.76, put 0.81), Hull's Greeks example (S=49, K=50, r=5%, sigma=20%, T=20 weeks) and Hull's implied-volatility example (call 1.875 gives about 23.5%). Hull values are published to two or three digits, so those tests use matching tolerances.
- Higher-precision checks (to about 1e-11) compare against an independent Python implementation built on `math.erfc`. These guard against implementation slips; they are not independent derivations of the formulas.
- The binomial pricer reproduces Hull's two-step American put (7.43) and five-step American put (4.49), and converges to BSM as steps increase (CRR oscillates, so convergence is not monotone).
- Monte Carlo is tested against the closed form within standard errors (fixed seeds, so results are deterministic) and Longstaff-Schwartz against the binomial tree. The random number generator is checked statistically (moments, tail mass), not against published test vectors.
- SVI fitting is tested on noise-free and noisy synthetic data generated from known parameters; the surface arbitrage checks are tested on constructed violations. The SVI butterfly density `g(k)` follows Gatheral and Jacquier (2014) and has not been checked against a second implementation.
- Property tests (proptest) cover put-call parity, monotonicity in spot, delta bounds, American >= European, no-arbitrage price bounds, implied-vol round trips, SVI round trips, Monte Carlo vs closed form, and VaR/CVaR/drawdown invariants.
- The normal CDF uses `libm::erfc` (about 1 ulp, with relative accuracy in the tails).

Limits: the CRR tree is `O(n^2)` and its Greeks are finite differences, so they are noisier than the analytic ones. Implied volatility is ill-conditioned for deep in-the-money options priced at intrinsic value; the solver returns a volatility that reprices the option, which need not equal the volatility you started from. Plain Monte Carlo is unreliable for deep out-of-the-money options (almost every path pays zero). Longstaff-Schwartz is biased slightly low. The surface arbitrage checks look at the quoted nodes only, and the SVI fit does not enforce the butterfly condition (check it with `SviParams::is_butterfly_free`).

This crate is provided for education and research, with no warranty, and is not investment advice.

## Roadmap

Done in 0.2: Monte Carlo, volatility surfaces, batch/parallel APIs, serde, Python bindings.

Not yet: exotic options (barrier, Asian, digital), stochastic volatility models (Heston, SABR), arbitrage-free surface calibration (SSVI), fixed income, portfolio optimisation, an MCP or tool server for LLM agents, a PyPI release, market data feeds.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
