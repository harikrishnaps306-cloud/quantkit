# Dependency advisory and licence review (24 October 2026)

Scope: what was run, what it found, and what it does not cover.

## Advisories

| check | tool | data | result |
|---|---|---|---|
| Rust, library crate (all features, dev-dependencies included), newest compatible lockfile, 90 crates | `cargo audit 0.22.2` | RustSec advisory-db, 1,352 advisories, fetched 2026-10-24 | no vulnerabilities and no warnings reported |
| Rust, Python-binding crate (pyo3 0.30, numpy 0.30), 36 crates | `cargo audit 0.22.2` | same | no vulnerabilities and no warnings reported |
| Rust, library crate, Rust 1.74 fallback lockfile, 80 crates | `cargo audit 0.22.2` | same | no vulnerabilities and no warnings reported |
| Python, NumPy 1.21.6 (the declared floor) | `pip-audit` (OSV/PyPI data) | live | **1 known vulnerability: PYSEC-2021-855, fixed in 1.22** |
| Python, NumPy 1.26.4 and 2.5.4 | `pip-audit` | live | none known |

The lockfile used for the audit is regenerated each time because `Cargo.lock`
is deliberately not committed for this library; users will resolve newer
releases than the audit saw. Re-run the audit before each release.

### NumPy 1.21.6 and PYSEC-2021-855

The advisory is a 2021 NumPy issue (the same family as CVE-2021-41495 and
CVE-2021-41496, which concern `numpy.f2py` and sorting and which NumPy
maintainers have disputed). quantkit does not call `f2py`, and its bindings use
NumPy only for array exchange, so the issue does not look reachable through
quantkit. That is an assessment from reading the advisory title and quantkit's
code surface, not a test. The consequence for the declared range is a decision
for the maintainer: either keep `numpy>=1.21.6` (maximum compatibility, with
the advisory documented here) or raise the floor to `1.22`, which also requires
changing the CI pin, `docs/compatibility.md` and re-running the matrix row.
No change was made.

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
