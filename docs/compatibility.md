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

## Executed: wheel in a clean virtualenv, 23 tests each (Windows 11 x86-64)

One wheel, `quantkit-0.2.0-cp39-abi3-win_amd64.whl` (built before the distribution was renamed to `quantkit-rs`; it is now `quantkit_rs-0.2.0-...`), built with
`maturin 1.15.0` from a fresh clone of commit `2b5e59a`, installed into a fresh
`uv venv` for each row (`include-system-site-packages = false`, checked in each
`pyvenv.cfg`); the test directory was copied out of the checkout and
`quantkit.__file__` was asserted to be inside `site-packages`. Rows 1, 4 and 5
first failed to import NumPy itself ("DLL load failed ... filename or extension
is too long") because the scratch directory path was very long; that is a
Windows path-length artefact of old NumPy builds, not a quantkit result. They
were rerun from a short path (`C:/qkmx`) and passed.

| # | CPython | NumPy | result |
|---|---|---|---|
| 1 | 3.9.25 | 1.21.6 | 23 passed |
| 2 | 3.9.25 | 1.26.4 | 23 passed |
| 3 | 3.9.25 | 2.0.2 | 23 passed |
| 4 | 3.10.20 | 1.21.6 | 23 passed |
| 5 | 3.11.9 | 1.23.5 | 23 passed |
| 6 | 3.11.9 | 1.26.4 | 23 passed |
| 7 | 3.11.9 | 2.4.6 | 23 passed |
| 8 | 3.12.13 | 1.26.4 | 23 passed |
| 9 | 3.12.13 | 2.5.4 | 23 passed |
| 10 | 3.13.14 | 2.5.4 | 23 passed |

Also executed: the source distribution built from the same commit (no build
outputs inside; licence files present) was built and installed by pip into a
fresh isolated virtualenv on CPython 3.12.13 with NumPy 2.5.4, and the same 23
tests passed against the installed copy.

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
| 1.97.1 (Windows), commit `2b5e59a`, fresh clone | version guard, fmt, clippy `-D warnings` (all targets, all features), tests with all features (171 = 123 library + 48 integration), 93 doctests, golden vectors serial and `parallel` at 1 and 7 threads | pass |
| 1.74.1 (Windows), same commit | tests with all features under the resolver-fallback lockfile (171), 93 doctests | pass |

Earlier passes ran the default, `serde` and `parallel` feature sets separately,
docs, benches, examples and `cargo package`; those were not repeated in the
second pass above.

The Python bindings (pyo3 0.30, numpy crate 0.30) are not covered by the
Rust 1.74 claim, which concerns the library crate only.
