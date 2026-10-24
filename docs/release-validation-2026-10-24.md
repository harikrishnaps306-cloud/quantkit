# Release validation, 24 October 2026

Scope: local validation of commit `2b5e59a` (the audit fixes) on Windows 11 x86-64 only. **No GitHub repository exists yet, so no CI has run, and nothing was tagged or published.**

## Executed

| check | result |
|---|---|
| fmt, clippy `-D warnings` (all targets, all features), `cargo doc` `-D warnings`, version check | pass |
| Rust 1.97.1 tests: default 170, serde 171, parallel 170, all features 171; 93 doctests | pass |
| Golden vectors, serial and parallel at 1 and 7 threads; no lock re-recorded | pass |
| Rust 1.74 (fresh clone, resolver-fallback lockfile): all targets 171, 93 doctests | pass |
| Wheel built from a fresh clone, installed in 10 isolated `uv venv` environments (no system packages), CPython 3.9-3.13 x NumPy 1.21.6-2.5.4, 23 tests each | 10/10 pass |
| sdist: built, 60 files, both licences present, no build outputs; installed and tested in a clean CPython 3.12 venv, 23 tests | pass |
| `cargo audit` 0.22.2, RustSec database of 1352 advisories, on a fresh latest-resolve lockfile (90 crates), the MSRV-fallback lockfile (80) and the Python crate (36) | no vulnerabilities or warnings reported |
| Licences of all 90 crates in the dependency graph | every crate offers MIT and/or Apache-2.0 (or Unlicense/BSD-2/Unicode-3.0 permissive terms); `r-efi` is `MIT OR Apache-2.0 OR LGPL-2.1-or-later` (a UEFI-target dependency; the permissive options apply). Details and the missing third-party-notices file: `docs/dependency-review.md`. Runtime dependencies: `thiserror`, `libm`, optional `serde` and `rayon`. |
| Package names | crates.io: `quantkit` is **free** (HTTP 404 on the API). PyPI: **`quantkit` is taken** by an unrelated 2021 project (`mmngreco/quantkit`, versions 0.0.1a1 and 0.0.1rc1). |

## Findings that need your decision

1. **PyPI name collision.** The wheel's distribution name is `quantkit`, which exists on PyPI. Publishing as is would fail or, worse, confuse users. The import name can stay `quantkit`; the distribution name in `python/pyproject.toml` would need to change (404 on PyPI at check time: `quantkit-options`, `qkit-py`, `quantkit-rs`). I did not rename anything.
2. **No repository, so no CI.** The Linux/macOS results, wheels for those platforms, and the cross-platform golden-vector check remain unobserved.
3. **Publishing** was not done: CI evidence does not exist, and the name issue is open.

## Not covered

Linux, macOS, ARM; Linux/macOS wheels (a Windows machine cannot build them; the `python` job is configured for them but unrun); `cargo deny` policy files; advisories for the test tooling (pytest, maturin). `pip-audit` was run on NumPy and **flagged PYSEC-2021-855 on the declared floor 1.21.6** (see `docs/dependency-review.md`; floor not changed).

## Benchmarks

Measured on the corrected code on an otherwise idle machine; see `docs/BENCHMARKS.md`. A first set taken while another benchmark process was running was discarded because the runs overlapped. The Python benchmark was not re-run.
