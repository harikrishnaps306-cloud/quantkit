# Dependency advisory and licence review (24 October 2026)

Scope: what was run, what it found, and what it does not cover.

## Advisories

| check | tool | data | result |
|---|---|---|---|
| Rust, library crate (all features, dev-dependencies included), newest compatible lockfile, 90 crates | `cargo audit 0.22.2` | RustSec advisory-db, 1,352 advisories, fetched 2026-10-24 | no vulnerabilities and no warnings reported |
| Rust, Python-binding crate (pyo3 0.30, numpy 0.30), 36 crates | `cargo audit 0.22.2` | same | no vulnerabilities and no warnings reported |
| Rust, library crate, Rust 1.74 fallback lockfile, 80 crates | `cargo audit 0.22.2` | same | no vulnerabilities and no warnings reported |
| Python, NumPy 1.21.6 (the former floor) | `pip-audit` (OSV/PyPI data) | live | 1 known vulnerability: GHSA-fpfv-jqm9-f5jm / PYSEC-2021-855 / CVE-2021-34141, fixed in 1.22 |
| Python, NumPy 1.22.0 and 1.22.4 (the new floor and a later 1.22) | PyPI JSON `vulnerabilities` field | live | none listed |
| Python, NumPy 1.26.4 and 2.5.4 | `pip-audit` | live | none known |

The lockfile used for the audit is regenerated each time because `Cargo.lock`
is deliberately not committed for this library; users will resolve newer
releases than the audit saw. Re-run the audit before each release.

### NumPy before 1.22 (resolved by raising the floor)

The advisory is GHSA-fpfv-jqm9-f5jm, with aliases CVE-2021-34141 and
PYSEC-2021-855: an incorrect string comparison in `numpy.core` that lets crafted
string objects make NumPy APIs fail (denial of service). Registries disagree on
the range. The GitHub advisory and PyPI list all releases before 1.22 as
affected, fixed in 1.22; the OSV PYSEC record lists only 1.9.0 to 1.9.3. The
more conservative range was followed. (An earlier version of this note described
the issue as an `f2py` problem; that was wrong.) quantkit passes float64 arrays
to NumPy, not attacker-built string objects, so exposure through quantkit looks
low, but that is an assessment, not a test.

**Resolution:** the declared minimum is now `numpy>=1.22.0`. 1.22.0 and 1.22.4
show no vulnerabilities on PyPI at check time. The wheel was rebuilt and the 23
tests pass on CPython 3.9 with NumPy 1.22.0 and on CPython 3.10 with NumPy 1.22.0
and 1.22.4. Metadata (`python/pyproject.toml`), CI pins, and
`docs/compatibility.md` were changed together.

### Limits of this scan

* Advisory databases list known issues only. "No advisories" is not a proof of
  absence of vulnerabilities, and unmaintained-crate warnings would have been
  printed and were not.
* Transitive dependencies of `pytest`, `maturin` and the test environment were
  not scanned; only NumPy, the one runtime dependency of the wheel, was.
* The audit ran on the newest compatible versions and, separately, on the Rust
  1.74 resolver-fallback lockfile (80 crates, older releases): no
  vulnerabilities or warnings in either.

## Licences

Source: `cargo metadata --all-features` for both manifests, licence expression
of every external package.

| set | external packages | licence expressions found |
|---|---|---|
| library crate, including dev- and target-specific dependencies | 79 | MIT, Apache-2.0, `MIT OR Apache-2.0` (most), `Unlicense OR MIT`, `Apache-2.0 WITH LLVM-exception` combinations, `BSD-2-Clause OR Apache-2.0 OR MIT` (zerocopy), `(MIT OR Apache-2.0) AND Unicode-3.0` (unicode-ident), `MIT OR Apache-2.0 OR LGPL-2.1-or-later` (r-efi) |
| Python-binding crate | 34 | the same families plus `BSD-2-Clause` (the `numpy` crate), `MIT` (libm), `Apache-2.0 WITH LLVM-exception` (target-lexicon) |

* No package lacks a licence expression, and none is GPL-only or MPL.
* `r-efi` offers LGPL-2.1-or-later only as one alternative alongside MIT and
  Apache-2.0, and it is used for UEFI targets, not for any platform quantkit
  builds for.
* `unicode-ident` carries `Unicode-3.0` for its data tables in addition to
  MIT/Apache-2.0; redistribution of binaries should include that notice.
* The shipped wheel statically links the Rust dependencies of the binding
  crate (34 packages). Their licences require preserving copyright and licence
  notices when redistributing binaries; the wheel does not currently bundle a
  third-party notices file. Producing one (for example with `cargo about` or
  `cargo-bundle-licenses`) before a public release is an open item.
* Licence expressions come from package metadata; the texts inside each
  package were not read.

This is an engineering review, not legal advice.
