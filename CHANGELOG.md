# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- Numerical Greeks now propagate pricer errors instead of treating them as parameter boundaries and silently switching stencils.
- Extreme CRR trees use log-domain rollback instead of silently losing terminal stock nodes to underflow. Non-finite results are errors.
- CVaR uses a scaled mean; Sharpe and Sortino normalize inputs before arithmetic. Representable extreme-scale ratios no longer silently become zero or NaN.
- Normal quantiles use log-tail refinement beyond eight standard deviations, including subnormal probabilities.
- Surface log-moneyness and reconstructed strikes avoid overflowing forwards. Indistinguishable log coordinates are rejected.
- Arbitrage reports expose `evaluation_errors`; `is_clean()` is false when evaluations fail. Python reports add `evaluation_errors` and `is_clean` keys. Adding the Rust field affects callers constructing report struct literals.
- Implied-volatility convergence errors carry actual absolute price residuals.
- Normalized Black price helpers reject non-finite/invalid arguments with NaN before evaluation. SVI uses `hypot` instead of squaring large coordinates.
- Python dataset metadata includes `spec`, containing the arguments needed to replay generation.

Validation of these changes is recorded in `docs/audit-fixes.md`. Earlier compatibility and performance reports describe earlier revisions; they are not new cross-platform evidence.

## [0.2.0]

Initial experimental release candidate (not published). The `0.x` API may change between minor versions.

Unreleased locally; nothing here has been published, so the "Changed" items are
differences from the earlier 0.2.0 development snapshot (commit `02a6393`), listed
so that code written against it can be migrated.

### Release preparation
- Added `docs/VERSIONING.md`: the `0.x` policy (breaking changes only in minor releases; what counts as breaking).
- The Python wheel and sdist now carry `THIRD_PARTY_NOTICES.txt` (licence texts of the 29 statically linked Rust packages), generated and checked in CI by `.github/scripts/gen_notices.py`.
- The Python distribution is renamed `quantkit-rs` because `quantkit` is taken on PyPI. The import name stays `quantkit`.
- The wheel's NumPy requirement is `numpy>=1.22.0` (it was `>=1.21`). NumPy before 1.22 is listed as affected by GHSA-fpfv-jqm9-f5jm / CVE-2021-34141; 1.22.0 is the oldest release tested and has no known advisory. The exact tested and configured Python/NumPy combinations are in `docs/compatibility.md`.
- CI now triggers on pushes to `main` or `master`, on pull requests and manually (`workflow_dispatch`); jobs for Rust on three operating systems, the Rust 1.74 check (library, tests, benches, examples, doctests), a Python 3.9/3.11/3.13 x NumPy matrix that tests the installed wheel, an sdist build-and-test job, and a version-consistency check. First green run on GitHub: commit `9a28f6e` (all 13 jobs on Linux, macOS and Windows).
- Compatibility evidence recorded in `docs/release-readiness.md`: the wheel on CPython 3.9-3.13 with NumPy 1.21.6-2.5.4 (Windows), Rust 1.74.1 for all targets, and an sdist install.
- The source distribution no longer contains build outputs, and both the wheel and the sdist carry the licence files (`python/LICENSE-*.txt` are copies of the root licences).
- `docs/RELEASE_NOTES_0.2.0.md` (draft, unpublished) and `docs/release-readiness.md`.

### Added
- `black` module: the normalised Black function evaluated stably far out of the money (`normalised_otm`, `ln_normalised_otm`, `ln_normalised_vega`).
- `limits`: the supported input domain, enforced by `OptionParams`, `Contract`, `VolSurface::new` and `dataset`.
- `StatusCode` and `QuantError::status_code()`; `detailed` module with per-row `PriceRow`, `GreeksRow`, `IvRow`, `McRow` and column-oriented `price_columns` / `greeks_columns`.
- `implied_vol::solve_detailed` returning `IvSolution` (residual, Newton/bisection step counts, vega).
- `mc::american_lsm_detailed` / `LsmResult`: independent training and evaluation paths, in-sample versus out-of-sample price, fit diagnostics. `mc::Rng::for_stream`, `mc::derive_seed`.
- `dataset` module: seeded, audited synthetic option datasets (`generate`, `DatasetSpec`, `Dataset`, `InvariantReport`, CSV export, checksum).
- `surface`: `check_arbitrage_sampled`, `ArbitrageScope`, vertical-spread (slope) condition, `SviFit::max_abs_residual`.
- Python: n-dimensional arrays with scalar pass-through, `*_detailed` functions, `STATUS`, `status_name`, `make_dataset`, `mc_american_detailed`, `set_num_threads`.
- `docs/NUMERICS.md`, `docs/BENCHMARKS.md`, `docs/market-gap-report.md`; throughput benchmarks; independent-reference, domain-boundary and no-panic test suites.

### Changed (breaking relative to the earlier snapshot)
- `bsm::price` uses the stable normalised-Black evaluation (relative error at `|d1| = 30` fell from 1.4e-10 to 1.5e-13); prices change in the last digits, so the dataset checksum regression lock was re-recorded.
- The implied-volatility solver is rewritten (log-domain Newton on the normalised Black function). Public signatures are unchanged. Prices as small as `1e-300` now solve instead of failing with `NoConvergence`.
- Inputs outside `quantkit::limits` are rejected (for example volatility below `1e-8`, spot above `1e12`, `|rate| * expiry > 500`).
- `bsm::price` clamps rounding-sized negative results to zero.
- Implied volatility: attainable-range failures are `PriceOutOfBounds` (previously some were `NoConvergence`); the documentation no longer claims unconditional convergence.
- Monte Carlo: results changed (block-based streams, `libm` functions); `std_error` for antithetic Longstaff-Schwartz is computed over pair means; `american_lsm` now returns the **out-of-sample** estimate trained on independent paths. European Monte Carlo and dataset results are thread-count independent and bitwise reproducible across platforms.
- `ArbitrageReport` gained a `scope` field; its documentation now states that clean reports are scoped.
- Risk: scale-safe standard deviations; results that would overflow are errors; max drawdown rejects equity paths that overflow.
- `greeks::numerical` rejects non-positive or non-finite bump sizes.
- Python: all-scalar batch calls return scalars (previously one-element arrays); array arguments must share a shape; `check_arbitrage()` returns an extra `scope` entry.
- Removed the guessed `repository` URL from `Cargo.toml`; set the real one before publishing.

### Correctness fixes in the latest review pass
- `mc::european` returned `Ok` with `NaN` when `S_T` overflowed but the discounted price did not (for example `r = 10`, `q = -10`, `T = 50`). It now simulates in discounted units divided by a common scale, and returns `NonFinite` only if the result itself is unrepresentable. American Monte Carlo returns `NonFinite` for overflowing simulated prices. Batch rows (`detailed`, columns) also turn any non-finite value into a non-`Ok` status.
- `greeks::numerical` accepted a positive bump that did not change the input (`delta = 0`, `gamma = NaN`). It now uses the actual floating-point steps, rejects steps below `1e-6` of the parameter's scale, uses second-order one-sided differences at the edges of the supported domain, and treats a non-finite pricer value as an error. **Breaking:** bumps below that scale, and the default absolute volatility/time bumps at magnitudes near 1000, are now rejected (pass explicit bumps).
- `mc::american_lsm_detailed` reported a `NaN` in-sample standard error from one effective observation. Training and evaluation now each need at least two independent observations (four antithetic paths). **Breaking:** `LsmResult::train_samples` / `eval_samples` count independent observations (antithetic pairs once); the new `train_paths` / `eval_paths` count paths.
- LSM estimator: the time-zero exercise decision is now made from the training paths and the fixed policy is then evaluated, instead of returning `max(evaluation mean, immediate payoff)` with the unfloored mean's standard error. **Breaking:** `price` / `std_error` follow the fixed policy; new fields `exercise_now`, `immediate_exercise_value`, `continuation_value`, `continuation_std_error`; `floored_price()` is a display helper, not an estimate. The documentation no longer claims the price is "slightly low" or that the in-sample value bounds the true value.
- `Rng::next_f64` could return exactly `1.0` (a 53-bit midpoint grid rounds up at its top end). It now maps the top 52 bits to midpoints `(k + 1/2) / 2^52`. **This deliberately changes every random stream's normals in the last bits**, so the Monte Carlo and dataset regression locks (Rust and Python) were re-recorded in the same change; the statistical, reference-vector and Black-Scholes agreement tests all pass, which is what justified the new values.
- Added reference-vector tests for `xoshiro256++` (the published vector for state `1, 2, 3, 4`) and `splitmix64`, independent of the statistical tests.
- Implied volatility: added `IvSolution::elasticity` (and `IvRow::elasticity`, Python `elasticity`) as the conditioning diagnostic; in-the-money prices whose time value is below `8 eps` times the price are now `PriceOutOfBounds`; the residual check has an absolute floor for subnormal prices. The documentation no longer says a price of `1e-300` is "as well conditioned as a price of 1".
- Documentation: stream-separation claims made consistent (distinct identifiers and states are not a proof of non-overlap); cross-platform reproducibility is stated for the one target actually checked; the American-equals-European identities now state their sufficient conditions (`q <= 0` and `r >= 0` for calls; `r <= 0` and `q >= 0` for puts) with tested counterexamples; zero-volatility and zero-expiry behaviour is described without overgeneralising; no-panic robustness is separated from numerical correctness.
- Release: `Cargo.lock` is no longer committed for the library crate; the CI workflow runs golden-vector tests on all three operating systems (configured, not yet executed); the Python wheel now ships the licence files.

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
