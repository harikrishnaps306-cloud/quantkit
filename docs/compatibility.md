# Compatibility: declared and actually tested

This is the single place that lists exactly which combinations were executed.
Other documents link here instead of restating them.

## Declared

| item | declaration | basis |
|---|---|---|
| Rust (library and dev-dependencies) | `rust-version = "1.74"` | passes on Rust 1.74.1 with a Rust-version-aware lockfile (see `docs/release-readiness.md`) |
| Python | `Requires-Python >=3.9` | one abi3 wheel tested on 3.9, 3.10, 3.11, 3.12, 3.13 (table below) |
| NumPy | `numpy>=1.22.0` | 1.22.0 is the **oldest release actually installed and tested**, and the first release without the known advisory GHSA-fpfv-jqm9-f5jm (see `docs/dependency-review.md`) |

**History of this bound.** It was `>=1.21` (only 1.21.6 had been tested), then
`>=1.21.6`, and is now `>=1.22.0` because every NumPy release before 1.22 is
listed as affected by GHSA-fpfv-jqm9-f5jm / CVE-2021-34141 (a denial-of-service
class "incorrect comparison" issue in `numpy.core` string handling). Releases
before 1.22.0 are therefore **unsupported**, not merely untested. The NumPy C API
is loaded at run time by the `numpy` crate, which documents `>= 1.16` as
recommended; that statement is not a test of quantkit.

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
| 1 | 3.9.25 | 1.21.6 | 23 passed (now **below the floor**; historical, unsupported) |
| 2 | 3.9.25 | 1.26.4 | 23 passed |
| 3 | 3.9.25 | 2.0.2 | 23 passed |
| 4 | 3.10.20 | 1.21.6 | 23 passed (now **below the floor**; historical, unsupported) |
| 5 | 3.11.9 | 1.23.5 | 23 passed |
| 6 | 3.11.9 | 1.26.4 | 23 passed |
| 7 | 3.11.9 | 2.4.6 | 23 passed |
| 8 | 3.12.13 | 1.26.4 | 23 passed |
| 9 | 3.12.13 | 2.5.4 | 23 passed |
| 10 | 3.13.14 | 2.5.4 | 23 passed |

Rows 11 to 13 use a wheel rebuilt after the floor change (metadata only; no
code changed), same method (`uv venv`, isolated, tests copied out of the checkout):

| # | CPython | NumPy | result |
|---|---|---|---|
| 11 | 3.9.25 | 1.22.0 | 23 passed |
| 12 | 3.10.20 | 1.22.0 | 23 passed (2 `DeprecationWarning`s raised inside NumPy's own `distutils` import) |
| 13 | 3.10.20 | 1.22.4 | 23 passed |

Rows 2, 3 and 5 to 10 were run against the earlier wheel and not repeated for
the rebuilt one; CI re-runs the oldest, a middle and the newest combinations.

Also executed: the source distribution built from the same commit (no build
outputs inside; licence files present) was built and installed by pip into a
fresh isolated virtualenv on CPython 3.12.13 with NumPy 2.5.4, and the same 23
tests passed against the installed copy.

## Executed on GitHub

Results are job and step conclusions read from the GitHub API; logs were not read
in full. Commits: first full pass `9a28f6e`; latest verified at the time of writing
`952203e` (CI 13 of 13, and the release dry run below). Later commits have their
own runs in the Actions tab.

The first run (commit `dcc3b18`) passed 12 of 13; the failure was the Ubuntu
CPython 3.9 row, because `ubuntu-latest` is now Ubuntu 26.04 and `setup-python`
has no 3.9 for it. That row is pinned to `ubuntu-24.04`.

### Which wheel was tested on which interpreter

Exactly these combinations, and no others, were installed and tested by the
workflows (a wheel is abi3, so one wheel per platform serves every CPython >= 3.9,
but only the listed interpreters were exercised):

| platform | CI (`ci.yml`, wheel built in the job) | Release dry run (`release.yml`, the artifact that would be uploaded) |
|---|---|---|
| Linux x86-64 | 3.9 + NumPy 1.22.0 (ubuntu-24.04), 3.11 + 1.26.4, 3.13 + newest | 3.9 + 1.22.0, 3.13 + newest |
| macOS arm64 | 3.11 + 1.26.4, 3.13 + newest | **3.13 + newest only** |
| Windows x86-64 | 3.9 + 1.22.0, 3.13 + newest | **3.9 + 1.22.0 only** |

So the macOS release artifact was tested on 3.13 only and the Windows release
artifact on 3.9 only; their other interpreters rest on the CI wheels (built the
same way) and on the abi3 design, not on a test of the release file. CPython 3.10,
3.12 and 3.14 were tested only locally on Windows (table above), and not at all
on Linux or macOS.

### Release workflow dry runs (publishing off)

Run 40125380060 on `952203e` (CI on the same commit also 13 of 13): three wheels
and the sdist built; each wheel installed and tested as in the table above; pip
refused NumPy 1.21.6 together with the Linux and Windows 3.9 wheels; `twine check
--strict` passed; `.github/scripts/inspect_dists.py` passed on all four files; the
PyPI job was skipped. Earlier dry runs: 40119810048 (`62bab38`, passed, before the
inspection steps existed) and 40124972016 (`100b390`, failed in the NumPy-floor
step on Windows because that virtualenv's bundled pip predated `--dry-run`; a
bug in the check, fixed in `952203e`).

Platform tags of the artifacts, as printed by the inspection step:

| file | tags |
|---|---|
| Linux | `cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64` |
| macOS | `cp39-abi3-macosx_11_0_arm64` |
| Windows | `cp39-abi3-win_amd64` (checked on the locally built wheel) |
| sdist | `quantkit_rs-0.2.0.tar.gz` |

So the Linux wheel needs glibc 2.17 or newer, and the macOS wheel runs on Apple
silicon with macOS 11 or newer; Intel Macs, Linux ARM and musl get no wheel.

## Not tested

* Linux and macOS (any interpreter), ARM, musl; the wheels built there.
* NumPy 1.22.1 to 1.22.3, and every NumPy release between the tested ones. Releases before 1.22.0 are unsupported (advisory).
* Combinations for which, to our knowledge, NumPy publishes no wheel: NumPy 1.22
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
