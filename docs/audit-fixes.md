# Quantkit audit fixes — 24 October 2026

Base: b1d4f702c20b503111eb60be781813e4d5baea0e. Changes are local and uncommitted. No remote, tag, or publication was created.

## Correctness fixes

- **CRR tree:** terminal-stock underflow previously erased positive nodes, returning zero for a call worth approximately 100 (S=K=100, volatility=100, T=1, 100 steps). An extreme-tree log-domain fallback preserves probability-weighted payoffs; non-finite final prices return errors. Ordinary trees retain the existing path. Extreme evaluation is slower.
- **Risk:** scaled means prevent opposite-sign overflow in CVaR; scaled Sharpe and Sortino calculations avoid false zero ratios or zero-downside errors. Overflowing standard deviations return errors. VaR rounding documentation was corrected.
- **Normal quantiles:** log-tail refinement handles subnormal probabilities without an overflowing inverse-density correction. Independent quantile reference tests cover the smallest positive f64 and other tails.
- **Surfaces:** log-forward coordinates avoid overflow and reproduce accepted quote nodes. Indistinguishable log coordinates are rejected. Reports now count failed price evaluations, and such reports are not clean.
- **SVI:** hypot avoids unnecessary overflow when squaring finite coordinates; second-derivative arithmetic uses normalized factors.
- **Implied volatility:** failure residuals are actual absolute model-price errors, not a log-error approximation.
- **Numerical Greeks:** callback pricing failures propagate; only invalid bumped parameters permit a one-sided fallback.
- **Black helpers:** invalid or non-finite price/log-price arguments return NaN before entering the evaluator.
- **Python datasets:** metadata now includes replayable generation arguments under `spec`.

## Compatibility notes

The Rust ArbitrageReport has a new public `evaluation_errors` field. Existing struct literals need that field or `..Default::default()`. Python arbitrage dictionaries add `evaluation_errors` and `is_clean`; dataset metadata adds `spec`. Numerical results can change on affected inputs. No existing Monte Carlo or dataset checksum locks were re-recorded.

## Verification

Executed on Windows x86-64 against workspace copies of the source:

- Stable Rust all-feature suite: 171 tests plus 93 doctests, including 12 new regression tests.
- Rust 1.74 all-feature suite and all-target compile check. Compatible dependencies were resolved online after an incomplete offline cache prevented the first attempt.
- Formatting, all-target/all-feature clippy with warnings denied, documentation with warnings denied, and version guard.
- Five robustness tests at 30,000 cases each. Successful risk/quantile results are now required to be finite; PROPTEST_CASES is explicitly honored.
- Rebuilt Windows abi3 wheel installed and tested: 23 Python tests on CPython 3.11.9 / NumPy 2.3.5. This virtualenv uses system-site-packages; it is not a fully isolated minimum-dependency test.
- Cargo package verification from a clean source copy.

The first five defects were reproduced before fixes. Additional tests check extreme-tree put-call parity and American exercise bounds, scale invariance, SVI finite output, invalid helper arguments, callback-error propagation, and Python dataset replay.

## Remaining limits

GitHub CI, Linux, macOS, ARM, cross-platform bitwise equality, and the full Python/NumPy compatibility matrix remain unverified for these changes. No new speed benchmarks or dependency-security advisory scan were run. The repository URL remains unset. An sdist build/install was not repeated in this pass.

This is finite-test evidence, not proof that every numerical input is correct. Public raw-f64 helpers and public SVI fields are not equivalent to validated OptionParams. Existing model limits remain: sampled surface checks are not global arbitrage guarantees, SVI fitting is not arbitrage-constrained, Monte Carlo confidence intervals describe sampling error, and volatility inference loses information near intrinsic or saturated prices.

Release readiness still depends on running actual multi-platform CI and reviewing the resulting platform artifacts.
