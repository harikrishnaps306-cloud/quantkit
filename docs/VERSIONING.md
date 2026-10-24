# Versioning and compatibility policy (0.x)

quantkit is an experimental `0.x` project. This page states when things may
change, so users can decide how tightly to pin. It describes intent for a small,
single-maintainer project; it is not a contract or a support commitment.

## One version number

The Rust crate (`quantkit`) and the Python distribution (`quantkit-rs`, import
name `quantkit`) are released together with the same version number.
`.github/scripts/check_versions.py` enforces this in CI.

## What the digits mean while the major version is 0

| change | version bump |
|---|---|
| breaking changes (see list below) | **minor**: `0.2.x` to `0.3.0` |
| bug fixes, documentation, new additive items, performance work that changes no documented result | **patch**: `0.2.0` to `0.2.1` |
| `1.0.0` | not planned yet; it would be announced in the changelog, and nothing here implies a date |

Breaking changes can therefore appear in any minor release, and **only** there.
Pin `quantkit-rs>=0.2,<0.3` (or `quantkit = "0.2"` in Cargo, which Cargo already
treats as `<0.3`) if you need stability within a series.

A patch release can still change numerical output when it fixes a bug. Such a
change is listed in the changelog under "Fixed" with the affected inputs.

## What counts as breaking

Any of these may change in a minor release:

* Public Rust or Python names, signatures, struct fields, enum variants and
  dictionary keys. Adding a field to a public struct or a variant to a public
  enum is breaking for code that builds or matches them exhaustively (this has
  already happened once: `ArbitrageReport::evaluation_errors`).
* Numerical results, beyond the accuracy statements in `docs/NUMERICS.md`. The
  stated accuracy is the contract; the last bits of an output are not.
* Seeded outputs: Monte Carlo prices and standard errors, and generated datasets
  and their checksums. A seed reproduces a result **within one version**; across
  versions it is reproducible only if the changelog says the streams did not change.
  The regression-locked values were deliberately re-recorded once during
  development.
* The supported input domain (`quantkit::limits`) and which inputs return which
  error or `StatusCode`.
* The minimum supported Rust version (`rust-version`), the minimum Python and
  NumPy versions, and the set of build targets with wheels.
* Cargo feature names and what they enable.
* The text of error messages (match on error kinds and status codes instead).

## What is least likely to change

These are the intended core. They will still change in a minor release if a
correctness problem requires it, with a changelog entry: `OptionParams::builder`,
`bsm::price`, `greeks::analytic`, `implied_vol::solve`, `binomial::price`,
`mc::european`, and `risk::*`. The `detailed` row types, `dataset`, `surface`
result types, `LsmResult`, `Bumps` defaults and the Python result dictionaries
are the most likely to move.

## How changes are communicated

* Every breaking change is listed in `CHANGELOG.md` under a "Changed" or
  "Removed" heading marked **breaking**, with migration notes where a one-line
  fix exists.
* Where practical, an item is deprecated for one minor release before removal.
  This is a goal, not a guarantee; a correctness or safety fix may remove or
  change something immediately.
* The changelog heading for an unpublished or yanked version says so.

## Supported toolchains

Only combinations in `docs/compatibility.md` are claimed. Raising the minimum
Rust, Python or NumPy version is a breaking change (minor bump) and the new
minimum must have been tested. Platforms not listed there (for example ARM
Linux, musl, free-threaded CPython, PyPy) are **unsupported**, not merely untested:
reports are welcome but nothing is promised.

## Security and correctness fixes

Fixes land in the newest minor series. Older series are not maintained; there
are no long-term-support branches. A release with a serious defect may be
yanked from crates.io or PyPI, with the reason recorded in the changelog.

## Not covered

No claim of fitness for trading, risk reporting or any regulated use is made;
see the licence for the warranty disclaimer, and `docs/NUMERICS.md` for what the
numerical methods do and do not guarantee.
