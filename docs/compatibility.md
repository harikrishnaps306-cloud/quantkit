# Compatibility: declared and actually tested

This is the single place that lists exactly which combinations were executed.
Other documents link here instead of restating them.

## Declared

| item | declaration | basis |
|---|---|---|
| Rust (library and dev-dependencies) | `rust-version = "1.74"` | passes on Rust 1.74.1 with a Rust-version-aware lockfile (see `docs/release-readiness.md`) |
| Python | `Requires-Python >=3.9` | one abi3 wheel tested on 3.9, 3.10, 3.11, 3.12, 3.13 (table below) |
| NumPy | `numpy>=1.21.6` | 1.21.6 is the **oldest release actually installed and tested** |

**Correction made in this pass.** The wheel previously declared `numpy>=1.21`.
Only NumPy 1.21.**6** had been tested; releases 1.21.0 to 1.21.5 never were, so
the declaration claimed more than the evidence supported and was raised to
`>=1.21.6`. This is not a finding that older 1.21 releases fail; it is that they
are untested. The NumPy C API is loaded at run time by the `numpy` crate, which
documents `>= 1.16` as recommended but that statement is not a test of quantkit.

## Executed: wheel in a clean virtualenv, 20 tests each (Windows 11 x86-64)

One wheel, `quantkit-0.2.0-cp39-abi3-win_amd64.whl`, built with
`maturin 1.15.0`, installed into a fresh virtualenv for each row; the test
directory was copied out of the checkout and `quantkit.__file__` was asserted to
be inside `site-packages`.

| # | CPython | NumPy | result |
|---|---|---|---|
| 1 | 3.9.25 | 1.21.6 | 20 passed |
| 2 | 3.9.25 | 1.26.4 | 20 passed |
| 3 | 3.9.25 | 2.0.2 | 20 passed |
| 4 | 3.10.20 | 1.21.6 | 20 passed |
| 5 | 3.11.9 | 1.23.5 | 20 passed |
| 6 | 3.11.9 | 1.26.4 | 20 passed |
| 7 | 3.11.9 | 2.4.6 | 20 passed |
| 8 | 3.12.13 | 1.26.4 | 20 passed |
| 9 | 3.12.13 | 2.5.4 | 20 passed |
| 10 | 3.13.14 | 2.5.4 | 20 passed |

Also executed: a wheel built **from the source distribution** installed on
CPython 3.11.9 with the newest NumPy available for it at the time (the exact
version was not recorded), 20 passed.

## Configured in CI, not yet executed

`.github/workflows/ci.yml`, job `python` (`numpy` with no pin means the newest
release for that interpreter):

| runner | CPython | NumPy |
|---|---|---|
| ubuntu-latest | 3.9 | 1.21.6 |
| ubuntu-latest | 3.11 | 1.26.4 |
| ubuntu-latest | 3.13 | newest |
| windows-latest | 3.9 | 1.21.6 |
| windows-latest | 3.13 | newest |
| macos-latest (arm64) | 3.11 | 1.26.4 |
| macos-latest (arm64) | 3.13 | newest |

plus job `sdist` (ubuntu-latest, CPython 3.12, newest NumPy).

## Not tested

* Linux and macOS (any interpreter), ARM, musl; the wheels built there.
* NumPy 1.21.0 to 1.21.5, and every NumPy release between the tested ones.
* Combinations for which, to our knowledge, NumPy publishes no wheel: NumPy 1.21
  with Python 3.11 or later, NumPy 1.x with Python 3.13. These were not attempted.
* Free-threaded CPython and PyPy (not claimed).

## Rust

| toolchain | what ran | result |
|---|---|---|
| 1.97.1 (Windows) | fmt, clippy, tests (default 158, serde 159, parallel 158, all features 159), 93 doctests, docs, benches, examples, `cargo package` | pass |
| 1.74.1 (Windows) | check and test, all targets, all features (159 = 123 library + 36 integration), 93 doctests | pass |

The Python bindings (pyo3 0.30, numpy crate 0.30) are not covered by the
Rust 1.74 claim, which concerns the library crate only.
