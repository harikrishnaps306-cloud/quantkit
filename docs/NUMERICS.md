# Numerical conventions and guarantees

This document states what quantkit computes, in which units, over which input
domain, and what is and is not guaranteed. Where a guarantee is "tested", the
test that checks it is named. Nothing here is a claim that the library is free
of defects.

## 1. Supported input domain

`OptionParams` rejects inputs outside the bounds in `quantkit::limits`:

| quantity | accepted |
|---|---|
| spot, strike | `[1e-8, 1e12]` |
| volatility | `[1e-8, 1e3]` (annualised, a fraction) |
| time to expiry (years) | `[1e-8, 1e3]` |
| risk-free rate, dividend yield | `[-10, 10]` and `abs(value) * expiry <= 500` |

Within that domain the closed-form price and all five analytic Greeks are
finite, the price is non-negative and below its no-arbitrage upper bound, and
put-call parity holds to `1e-9` of the discounted-asset scale
(`tests/domain.rs` sweeps the boundary and interior values of every parameter,
over 10,000 accepted combinations, and checks this). Outside it you get an error, not a
`NaN`.

**Zero volatility and zero expiry are rejected, not approximated.** At those
points Black-Scholes degenerates to the discounted forward intrinsic value;
gamma and vega are undefined or discontinuous there, and quantkit does not
manufacture finite numbers for them. Near the boundary the price converges to
that limit (`smallest_supported_vol_gives_forward_intrinsic_value`), so for a
deterministic payoff compute `max(S e^{-qT} - K e^{-rT}, 0)` directly.

Rounding can make the difference of two nearly equal terms slightly negative
(about `1e-16` of the term size); the BSM price is clamped to `0.0` in that
case.

## 2. Conventions

* Continuous compounding, continuous dividend yield `q`, forward
  `F = S e^{(r-q)T}`, discount factor `e^{-rT}`.
* **Delta** `dV/dS`. **Gamma** `d2V/dS2`.
* **Vega** `dV/d(sigma)` per **1.00** of volatility (divide by 100 for per
  vol point).
* **Theta** `dV/dt = -dV/dT` per **year** (divide by 365 for per day). It is
  usually negative for long options.
* **Rho** `dV/dr` per **1.00** of rate (divide by 100 for per percentage
  point). Dividend-yield sensitivity (epsilon) is not provided.
* The analytic Greeks are checked against 5-point finite differences of
  *prices only* (`tests/references.rs`), so they do not rely on the formulas
  they test.

## 3. American options on the binomial tree

* Cox-Ross-Rubinstein: `u = e^{sigma sqrt(dt)}`, `d = 1/u`,
  `p = (e^{(r-q) dt} - d) / (u - d)`. The probability must lie in `(0, 1)`;
  otherwise the call is rejected with an error (typically too few steps for a
  large `|r-q|` and small volatility).
* Early exercise is allowed at the `steps + 1` tree nodes' dates, i.e. this is
  a **Bermudan option with `steps` exercise dates** that approaches the American
  value as `steps` grows. Steps are limited to 100,000 (`O(steps^2)` work).
* The price is not monotone in `steps` (CRR oscillates); convergence is
  tested as a tolerance, not as monotonicity.
* Exact structural properties that are tested: with `r = 0` an American put
  equals the European put; with `q = 0` an American call equals the European
  call; and a long-dated American put approaches McKean's closed-form
  perpetual-put value from below (`tests/references.rs`).
* **Numerical Greeks** (`greeks::numerical`, `binomial::greeks`) are central
  differences with bumps (`spot 0.5%`, `vol 0.001`, `rate 1e-4`,
  `time 1/365`, shrunk to stay valid; all must be positive and finite). A CRR
  price is not smooth in spot at the scale of the node spacing, so
  `binomial::greeks` widens the spot bump to four node spacings (max 10%).
  Expect gamma to be noisy; the analytic European Greeks are the accurate
  reference.

## 4. Implied volatility

`implied_vol::solve_detailed` brackets the root in the supported volatility
range, then runs Newton with a bisection fallback. Guarantee:

* **`Ok(sol)`**: `|BSM(sol.vol) - price| <= max(tolerance, 8 eps) * price`, and
  `sol.vol` is inside the supported range. `sol.residual` is that absolute
  error. This is what is tested for *any* price in the no-arbitrage interval
  (`iv_solution_honours_its_documented_guarantee`).
* **Not guaranteed**: that `sol.vol` equals the volatility that generated the
  price. Where vega is small many volatilities reprice equally well; the
  volatility error is about `residual / vega`, and `sol.vega` is returned so
  you can judge it (`iv_round_trip_error_is_bounded_by_residual_over_vega`).
* **`Err`**: `PriceOutOfBounds` (outside the no-arbitrage range, or outside
  what the supported volatility range can produce) or `NoConvergence` (carries
  the last residual and iteration count). A price the model cannot resolve in
  double precision, such as a deep out-of-the-money call worth `1e-300`,
  fails visibly instead of returning an arbitrary number.
* Default tolerance `1e-12` relative to the price; default budget 100 steps.

## 5. Monte Carlo

* Risk-neutral GBM, exact simulation, constant-rate discounting.
* `std_error` is the **sampling** standard error of the mean (antithetic pairs
  are averaged first). It does **not** include model, discretisation, or
  (for American options) exercise-policy error. A normal confidence interval
  from it is reliable for thousands of paths on non-extreme payoffs; plain
  Monte Carlo is unreliable for deep out-of-the-money payoffs where almost
  every path pays zero.
* **Reproducibility.** Results are a pure function of
  `(inputs, paths, seed, antithetic, steps)` and the crate version. They do not
  depend on the thread count (work is split into fixed 4096-sample blocks, each
  with its own stream from `Rng::for_stream(seed, block)`, merged in block
  order) and are bitwise identical across platforms, because the generator,
  `exp`, `log`, `sin`/`cos` and the normal CDF use pure-Rust `libm` and
  IEEE-exact operations. Locally verified: serial, `parallel` with 1 and 7
  threads, debug and release, and Rust 1.74 all give identical bits
  (`european_bitwise_regression_lock`). Other operating systems are covered by
  CI configuration that has not yet run.
* Streams for different `(seed, stream)` pairs come from independently hashed
  seeds: overlap is astronomically unlikely but not *proved* impossible (no
  jump-ahead). Distinct blocks and the training/evaluation domains were
  checked for output collisions (`streams_do_not_collide`).
* The generator is checked statistically (moments, tail mass), not against
  published test vectors.

### Longstaff-Schwartz

* Regression of discounted continuation cash flows of in-the-money paths on
  `1, z, z^2`, `z = (S/K - mean)/sd` (standardised per date, minimum 10
  in-the-money paths, singular fits counted in `regression_failures`; a date
  without a fit means "never exercise early" there).
* **Training and evaluation use independent paths** from disjoint stream
  domains, even if the two configs share a seed. The out-of-sample price is
  the value of a *feasible* stopping rule, so it is biased low relative to the
  Bermudan value, and it comes with a valid sampling standard error. The
  in-sample price reuses the regression's paths and is biased high; the gap
  is reported as a diagnostic. Test: over 200 small samples the in-sample
  estimate exceeds the independent estimate by more than 3 standard errors
  of the gap, and the independent estimate does not exceed the converged tree
  value beyond sampling error (`lsm_in_sample_is_biased_high_and_out_of_sample_is_not`).
* `price` is floored at the immediate-exercise payoff (exercising now is a
  feasible rule); `std_error` is for the unfloored mean.
* Error sources are distinct: **sampling** (`std_error`), **policy** (not
  measured; visible only through the in-sample/out-of-sample gap and by
  comparing with a tree), **discretisation** (`steps` dates), **model**
  (GBM).

## 6. Volatility surfaces and SVI

* Smiles are stored against forward log-moneyness `k = ln(K / F(T))`. Within a
  smile total variance `w = sigma^2 T` is interpolated linearly in `k`; across
  expiries linearly in `T` at fixed `k`; flat in volatility outside the
  quoted range. These are simple rules, not a calibrated model.
* **Arbitrage checks are scoped.** `check_arbitrage` examines the **quoted
  nodes** only (calendar: total variance non-decreasing in expiry at shared
  `k`; call prices non-increasing, slope-bounded by the discount factor, and
  convex in strike). `check_arbitrage_sampled` examines a finite grid of the
  **interpolated** surface. Reports carry their `scope`. A clean report is
  never a proof of a globally arbitrage-free surface.
* SVI: raw parameters with `b >= 0`, `|rho| < 1`, `sigma > 0`, non-negative
  minimum variance. `fit_svi` reports `rmse` and `max_abs_residual` and fails
  with `NoConvergence` if no admissible fit is found; it does **not** enforce
  the butterfly condition (use `SviParams::is_butterfly_free`). The butterfly
  density `g(k)` is checked against the density obtained by differentiating
  Black call prices twice with 5-point stencils
  (`svi_butterfly_g_matches_the_density_from_call_price_derivatives`).
* SSVI is intentionally not implemented: its parameter constraints and
  validation were out of scope for a rigorous implementation in this release.

## 7. Risk metrics

See the module documentation of `quantkit::risk` (simple returns, positive
loss convention, historical VaR rounding of the tail up, annualisation
assumptions, per-period risk-free rate and Sortino target, drawdown wealth
path). Successful results are finite; overflow is an error.

## 8. Platform and determinism scope

Bitwise-reproducible across platforms (pure `libm` and IEEE operations):
the `mc` module, `dist`, `bsm`, `greeks`, and `dataset`. Not guaranteed
bit-identical across platforms: the binomial pricer, implied-volatility solver,
`risk` and `surface` modules, which use the standard library's `exp`/`ln`.
Differences there are at the level of last-place rounding.

## 9. Python semantics

* Array arguments of any dimension must share a shape; scalars broadcast; an
  all-scalar call returns a scalar. Non-contiguous and Fortran-ordered arrays
  are accepted (values are read in logical row-major order). Empty arrays give
  empty results. Booleans (`is_call`) are scalars.
* `errors='raise'` raises `QuantError` naming the first failing row;
  `errors='nan'` returns `NaN` for failed rows. The `*_detailed` functions
  always return a `uint8` status per row (`quantkit.STATUS`), the failure
  message, and algorithm diagnostics.
* The GIL is released during computation. Batch work uses a global Rayon pool;
  set `RAYON_NUM_THREADS` or call `quantkit.set_num_threads(n)` before first
  use to avoid oversubscribing CPUs when you also parallelise in Python.
