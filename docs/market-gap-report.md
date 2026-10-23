# Market-gap report

Prepared 2026-10-23 from public pages read on that date. This is a small,
time-boxed scan, not a market study. "I did not find it" is recorded as that,
never as "it does not exist". Feature lists are taken from each project's own
README or package page and are **not independently verified**; benchmark
figures quoted from other projects are their authors' claims, not reproduced
here.

## What was checked

| project | what the page says it offers | source (accessed 2026-10-23) |
|---|---|---|
| RustQuant (Rust) | Modules for autodiff, instruments (bonds, options, pricing), stochastic processes, math, portfolio, data, ML basics. The README text does not list implied volatility, American options or batch APIs (the README may be incomplete). Dual MIT/Apache; about 1.8k stars, 26 open issues at the time of reading. | <https://github.com/avhz/RustQuant> |
| optionrs 0.2.1 (Rust) | Black-Scholes with Greeks and implied vol; binomial (European/American); Monte Carlo (European and exotics); exotics; PDE. Batch pricing not mentioned. Released 2025-12-21. | <https://docs.rs/crate/optionrs/latest> |
| quantrs 0.1.8 (Rust, with Python bindings) | Black-Scholes, Black-76, lattice, Monte Carlo, finite difference; Greeks for BS/Black-76; implied vol for Black-Scholes; American only on the lattice (Monte Carlo least squares listed as unsupported). Batch processing not mentioned. Released 2026-02-22. | <https://docs.rs/crate/quantrs/latest> |
| py_vollib_vectorized (Python) | Wraps py_vollib to accept arrays/pandas; numba is a required dependency; implied-vol function takes `on_error` of `raise`, `warn` or `ignore`; the page documents no per-row diagnostics (iterations, residuals). | <https://pypi.org/project/py-vollib-vectorized/>, <https://py-vollib-vectorized.readthedocs.io/en/latest/pkg_ref/iv.html> |
| pyvolr (Python, Rust core) | Black-Scholes-Merton and Black-76, Greeks with documented units, implied vol via Let's Be Rational, NumPy broadcasting with scalar-in/scalar-out, Rayon parallelism, abi3 wheels for Python 3.10-3.14; failed implied-vol rows give NaN with a warning (or raise, or ignore); American options and eSSVI listed as "coming soon". Self-reported benchmarks (not verified); 5 stars at time of reading. | <https://github.com/yipjunkai/pyvolr> |
| QuantLib (Python bindings) | Full C++ library; version 1.44 released 2026-10-13; abi3 wheels for CPython 3.9 plus free-threaded builds. A Monte Carlo American engine (Longstaff-Schwartz) exists in the C++ library; calibration-sample parameters are exposed in the engine (exact semantics not confirmed from official docs in this scan). | <https://pypi.org/project/QuantLib/>, <https://quantlib.js.org/docs/classes/_ql_pricingengines_mclongstaffschwartzengine_.mclongstaffschwartzengine.html> |
| ML option-pricing datasets (literature) | Papers generate synthetic Black-Scholes/Heston datasets by hand (sample S, K, T, r, sigma; price with a closed form or numerical method; sometimes add noise). The scan found no reusable seeded generator package. | e.g. <https://arxiv.org/pdf/2409.18168>, <https://arxiv.org/pdf/2510.01446>, <https://arxiv.org/pdf/2604.19832> |

## Candidate gaps, evidence and fit

| candidate | closest alternatives | specific limitation or unmet workflow | evidence of demand | confidence | fit / cost / maintenance |
|---|---|---|---|---|---|
| **A. Per-row batch diagnostics** (status code, error, solver residual/iterations/vega, Monte Carlo standard error per row) | pyvolr, py_vollib_vectorized | Documented failure handling is NaN plus a warning/exception (`on_error`); no per-row reason or solver diagnostics are documented on the pages read. The reason a row is NaN cannot be recovered from the output. | None direct. Indirect: two separate vectorised wrappers exist and both added an `on_error` switch, so failed rows are a known annoyance. | Documented gap in *documentation*; whether users need richer output is an **unvalidated hypothesis**. | Excellent: sits on the existing validation and error types; moderate code; low maintenance. **Selected.** |
| **B. Reproducible ML dataset generation** (seeded sampling, boundary corners, metadata, invariant report, checksum) | Hand-written scripts in papers | No package found that provides a seeded, audited generator with provenance metadata. | Indirect only: several papers each rebuild the same pipeline. No evidence anyone wants a library for it. | **Unvalidated hypothesis**. | Good: reuses batch, RNG and Greeks; small and self-contained; low maintenance. **Selected**, scoped to closed-form European labels. |
| Independent train/evaluation paths for American Monte Carlo | QuantLib (has calibration-sample parameters) | Not a gap versus QuantLib. quantrs lists LSM as unsupported and optionrs lists American pricing for the lattice. | n/a | Integration opportunity (done as a **correctness fix**, not as a differentiating feature) | Already implemented as part of fixing in-sample bias. |
| Faster / more robust implied volatility (Let's Be Rational) | pyvolr | pyvolr documents Jäckel's method with 1e-13 accuracy in at most two iterations and claims tail accuracy to prices near 1e-215; quantkit's Newton/bisection solver fails (visibly) near 1e-300 and is slower per row. | pyvolr exists and positions itself on this. | Documented competitor strength. | Real gap **against quantkit**, not a gap in the market. Deferred; see priorities. |
| Pricing snapshots (inputs, conventions, settings) | none found | Datasets from feature B carry spec, version, conventions and a checksum; a general snapshot format for arbitrary pricing runs was not built. | None found | Unvalidated hypothesis | Deferred. |
| Easy installation / NumPy interoperability | pyvolr (abi3 wheels), QuantLib (abi3 wheels) | Table stakes. quantkit builds abi3 wheels but none are published; CI to build them is configured and has not run. | n/a | Parity item | Release engineering, not a feature. |

## Selection

Two features were selected for this iteration: **A (per-row diagnostics)**
and **B (reproducible dataset generation)**. Neither rests on evidence of user
demand; both rest on (1) a documentation-level difference from the closest
tools, (2) clean architectural fit, and (3) low cost. If either sees no use,
it can be removed without touching the pricing core. Speed is deliberately not
the differentiator: a Rust-core competitor (pyvolr) already targets
vectorised Black-Scholes and implied-volatility throughput.

Not pursued: live data, trading, an LLM tool server, bonds, portfolio
optimisation, SSVI, additional models.

## Limits of this scan

* Pages were read once; READMEs can lag behind code. A feature missing from a
  README may exist.
* GitHub issue trackers were not mined for demand signals; one search for
  vectorised-IV error-handling issues returned no relevant issues.
* No competitor was installed or benchmarked.
