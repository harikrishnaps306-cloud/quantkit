# Release-readiness report (quantkit 0.2.0)

Written 2026-10-23 for the initial experimental release. It records what was
actually executed, on what, with what result, and what still depends on someone
or something outside this machine. Nothing was pushed, tagged or published.

## 1. Repository state

| item | value |
|---|---|
| branch | `master` (not renamed) |
| remotes | none configured |
| repository URL | **not set**: no verified URL exists, none was supplied, and the git `user.name` (`harikrishnaps306-cloud`) is not evidence of one, so none was invented. `cargo package` warns "manifest has no documentation, homepage or repository". |
| validated commit | `126a8a5` (the release-candidate commit; this report and a `.gitignore` tweak are committed after it) |
| working tree when validated | clean; all CI-equivalent steps below ran in a **fresh `git clone`** of that commit, so untracked local files cannot have influenced them |

## 2. What triggers CI after this change

`.github/workflows/ci.yml` previously ran on pushes to `main` only, but the local
branch is `master`, so nothing would have run. It now runs on:

* a **push to `main` or `master`** (either default-branch name works; no rename needed);
* **any pull request**;
* a **manual run** (`workflow_dispatch`, from the Actions tab or `gh workflow run ci.yml --ref <branch>`), which is the way to validate a feature branch before opening a pull request.

It does not run on tag pushes or on pushes to other branches. A new run on the
same ref cancels the previous one (`concurrency`).

Jobs: `metadata` (versions agree), `rust` (Linux, macOS, Windows), `msrv`
(Rust 1.74), `python` (seven OS/Python/NumPy combinations, each testing the
installed wheel), `sdist` (Linux).

## 3. Environment used

Windows 11 (10.0.26300), x86-64; Intel Core i9-14900HX; 15.7 GB RAM.

| tool | version |
|---|---|
| rustc / cargo (stable) | 1.97.1 |
| rustc / cargo (MSRV) | 1.74.1 |
| maturin | 1.15.0 |
| uv | 0.11.21 (used only to fetch interpreters and make virtualenvs) |
| pytest | 9.1.1 |
| CPython | 3.9.25, 3.10.20, 3.12.13, 3.13.14 (uv-managed), 3.11.9 (system) |
| NumPy | 1.21.6, 1.23.5, 1.26.4, 2.0.2, 2.4.6, 2.5.4 |
| Rust dependencies (MSRV lockfile) | libm 0.2.16, thiserror 1.0.69, serde 1.0.229, rayon 1.10.0, proptest 1.8.0, criterion 0.5.1 |
| Python crate dependencies | pyo3 0.30.1, numpy (crate) 0.30.0, ndarray 0.17.2, rayon 1.12.0 |

The stable-toolchain test runs used whatever the resolver picked at the time
(newer rayon, for example); the dependency versions that matter for the MSRV
claim are the ones in the table.

## 4. Checks executed

All in the fresh clone unless stated. Test counts differ by feature set and by
test target; they are not comparable across rows.

### Rust (stable 1.97.1), mirroring the `rust` job

| step | command | result |
|---|---|---|
| format | `cargo fmt --all -- --check` | pass |
| lint | `cargo clippy --all-targets --all-features -- -D warnings` | pass |
| tests, default | `cargo test --all-targets` | 158 passed |
| tests, serde | `... --features serde` | 159 passed |
| tests, parallel | `... --features parallel` | 158 passed |
| tests, all features | `... --all-features` | 159 passed |
| doctests | `cargo test --doc --all-features` | 93 passed |
| golden vectors, serial | `cargo test --lib -- xoshiro256pp splitmix64 unit_interval european_bitwise checksum_regression` | 5 passed |
| golden vectors, 1 and 7 threads | `RAYON_NUM_THREADS={1,7} cargo test --lib --features parallel -- european_bitwise checksum_regression` | 2 passed each |
| docs | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` | pass |
| benches | `cargo bench --no-run --all-features` | compile |
| examples | `cargo run --example <each of 4>` | all run |
| version check | `python .github/scripts/check_versions.py` | consistent (0.2.0 in both crates, the dependency, and the changelog) |
| package | `cargo package --list`, `cargo package` | pass; the packaged source is built and verified; 156 KB |

(From the working tree before the clone: `RAYON_NUM_THREADS`-varied golden
vectors, 8 repeated runs of the property/reference tests with random seeds, and
the no-panic suite at 30,000 cases per test also passed; those are not part of
the CI workflow.)

### Rust 1.74.1 (declared minimum), mirroring the `msrv` job

Lockfile resolved by `CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo +stable generate-lockfile` (lockfile `version = 3`):

| step | result |
|---|---|
| `cargo +1.74 check --lib` | pass |
| `cargo +1.74 check --all-targets --all-features` | pass (library, integration tests, benches, examples, **dev-dependencies**) |
| `cargo +1.74 test --all-targets --all-features` | 159 passed (123 library + 36 integration) |
| `cargo +1.74 test --doc --all-features` | 93 passed |

So both the library and its dev-dependencies build and pass on 1.74.1 when
resolved this way. The golden vectors (PRNG reference vector, Monte Carlo and
dataset locks) give identical results under the older compiler.

**Reproducing the MSRV check (maintainers):**

```bash
rustup toolchain install 1.74 --profile minimal
rm -f Cargo.lock    # the lockfile is not committed
CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo +stable generate-lockfile
cargo +1.74 test --all-targets --all-features && cargo +1.74 test --doc --all-features
```

Without the resolver setting, a plain `cargo generate-lockfile` can select
releases of rayon (and others) that need a newer compiler and the check will
fail for reasons unrelated to quantkit's own code. Removing `Cargo.lock` from
the repository does not by itself prove MSRV compatibility; the resolver
setting does the work. The Python bindings (pyo3 0.30) are **not** covered by
the Rust 1.74 claim.

### Python, mirroring the `python` and `sdist` jobs

One abi3 wheel (`quantkit-0.2.0-cp39-abi3-win_amd64.whl`) built from the clone with
`maturin build --release --manifest-path python/Cargo.toml -o dist`. For every
row, a clean virtualenv was created, the wheel, pytest and the pinned NumPy were
installed, the tests were copied out of the checkout, `quantkit.__file__` was
asserted to be under `site-packages` (so the installed wheel, not a source tree
or stale extension, was imported), and the 20-test suite was run.

| Python | NumPy | result |
|---|---|---|
| 3.9.25 | 1.21.6 (declared lower bound, the oldest tested) | 20 passed |
| 3.9.25 | 1.26.4 | 20 passed |
| 3.9.25 | 2.0.2 | 20 passed |
| 3.10.20 | 1.21.6 | 20 passed |
| 3.11.9 | 1.23.5 | 20 passed |
| 3.11.9 | 1.26.4 | 20 passed (also via the literal `python -m venv` steps of the CI job) |
| 3.11.9 | 2.4.6 | 20 passed |
| 3.12.13 | 1.26.4 | 20 passed |
| 3.12.13 | 2.5.4 | 20 passed |
| 3.13.14 | 2.5.4 | 20 passed |

Not attempted because no such wheels exist: NumPy 1.21 on Python 3.11 or later,
NumPy 1.x on Python 3.13. NumPy releases between the tested ones are assumed to
work. The suite covers scalars and arrays, non-contiguous and Fortran-order
arrays, empty arrays, shape errors, `raise`/`nan` modes, the detailed status
outputs, the updated `mc_american_detailed` and `implied_vol_detailed` fields,
the dataset generator, and the golden values.

Source distribution: `maturin sdist` builds, contains the sources and both
licence files and no build outputs (checked by pattern), installs in a clean
virtualenv (building the extension from source), imports from `site-packages`,
and passes the 20 tests (Python 3.11, NumPy 2.x).

Wheel metadata: `Requires-Python >=3.9`, `Requires-Dist: numpy>=1.21.6` (corrected from `>=1.21`, which had only been tested at 1.21.6; see `docs/compatibility.md`),
`Metadata-Version 2.4`; the wheel contains `licenses/LICENSE-MIT.txt` and
`licenses/LICENSE-APACHE.txt`, and so does a wheel built from the sdist. (The
`.txt` suffix on the copies in `python/` avoids a name collision with the root
licence files inside an sdist, which maturin rejects.)

## 5. Compatibility matrix

The exact executed and configured combinations are listed in [compatibility.md](compatibility.md).

| area | passed | failed | untested |
|---|---|---|---|
| Rust 1.97.1, Windows x86-64 | everything above | none | |
| Rust 1.74.1, Windows x86-64 | library, tests, benches, examples, doctests | none | |
| Rust, Linux / macOS / ARM | | | all (CI configured, not run) |
| CPython 3.9-3.13 x NumPy 1.21.6-2.5.4, Windows x86-64 | 10 combinations | none | intermediate NumPy releases |
| Python/NumPy on Linux / macOS | | | all (CI configured, not run) |
| free-threaded or PyPy builds | | | not claimed |
| bitwise reproducibility on other targets | | | all |

## 6. Workflow validation: what is and is not established

Established: `actionlint` 1.7.12 reports no errors for the workflow (it checks job and step structure, action input names, expression syntax and matrix references; `shellcheck` was not installed, so the embedded shell snippets were not shellchecked); and the commands inside every job were run locally, from a fresh clone,
with the shell, working directories, artifact paths and the "import the
installed wheel" assertion exercised on Windows with Git Bash. The workflow file
also parses as YAML and declares the triggers described above.

**Not established:** that the workflow behaves on GitHub. (`actionlint` cannot tell whether an action tag such as `dtolnay/rust-toolchain@master` or `actions/setup-python@v5` exists or behaves as assumed.) Specifically untested
are the action versions and inputs (`actions/checkout@v4`,
`dtolnay/rust-toolchain@stable` and `@master` with `toolchain: "1.74"`,
`Swatinem/rust-cache@v2`, `actions/setup-python@v5`), runner images,
the macOS arm64 and Linux paths of the shell snippets (for example
`.venv-clean/bin/python`), caching, and the older-Python/older-NumPy wheels on
Linux and macOS runners. Parsing the YAML proves none of this.

## 7. Environmental warnings seen

* git: "LF will be replaced by CRLF" for many files (Windows `core.autocrlf`); cosmetic here, but a fresh Linux checkout will have LF endings, which is the repository's intended form.
* uv: "Missing expected target directory for Python minor version link" while installing CPython 3.13; the interpreters were nevertheless installed and used.
* `cargo package`: missing `repository`/`documentation`/`homepage` metadata (see section 1).
* Path length: very long scratch paths broke a Windows linker step once; the fresh-clone runs used short paths.

## 8. Remaining release blockers

1. **CI has never run on GitHub.** Real Linux and macOS results are the main missing validation. Until they exist, nothing here is validated off Windows.
2. **No repository, so no URL.** Needs a real repository before `repository` (and `homepage`/`documentation`) can be set in `Cargo.toml` and `python/pyproject.toml` (`[project.urls]`).
3. **Reproducibility is unverified off one target.** The golden vectors have only been compared on Windows x86-64 with two compilers.
4. **Wheels for other platforms do not exist.** Only a Windows wheel was built; Linux and macOS wheels (and manylinux compliance) need CI or cibuildwheel, and the `numpy>=1.21.6` bound needs confirming there.
5. **Accuracy is evidence, not proof;** the accurate pricer and solver are slower than a naive implementation (see `docs/BENCHMARKS.md`); several reference figures from a textbook are unconfirmed and unused.
6. **Maintainer decisions not made:** crate/package name availability (checked 2026-10-24: `quantkit` is free on crates.io but **already taken on PyPI**, see `docs/release-validation-2026-10-24.md`); the final `0.x` API-stability policy (draft in `docs/RELEASE_NOTES_0.2.0.md`); whether to publish at all.

## 9. The exact next external action

Nothing further can be validated locally. Everything below is prepared and
**has not been run**; no remote exists and nothing has been pushed.

### What I need from you

1. A GitHub repository, **empty** (no README/licence/.gitignore, so the first push is a plain fast-forward), as `https://github.com/OWNER/NAME`, and whether it is **public or private**.
2. Explicit authorization to **push the local `master` branch** to it (only that branch; no tags).
3. How the push can authenticate from this machine. `gh` is not installed; a plain `git push` works if a Git credential helper already holds your GitHub login, otherwise you would run the push yourself or give me a way to authenticate. I will not ask you to paste a token into chat.

### What I will do, in order, once I have them

```bash
# 0. confirm the repository exists and is empty (read-only)
git ls-remote https://github.com/OWNER/NAME

# 1. record the verified URL in the metadata (dry run first, then write)
python .github/scripts/set_repository_url.py https://github.com/OWNER/NAME
python .github/scripts/set_repository_url.py https://github.com/OWNER/NAME --write
python .github/scripts/check_versions.py && cargo package     # still consistent and packageable
git commit -am "Set verified repository URL"

# 2. push master (this triggers CI: a push to master)
git remote add origin https://github.com/OWNER/NAME
git push -u origin master
```

No tag is created and nothing is published to crates.io or PyPI.

### How I will inspect the first run

* **Public repository:** the unauthenticated GitHub REST API lists runs, jobs and per-step conclusions (`https://api.github.com/repos/OWNER/NAME/actions/runs`, then `.../runs/<id>/jobs`), and the Actions page can be read in a browser. Raw log downloads need authentication even for public repositories, so for failures I would work from step names, conclusions and annotations and ask you to paste the failing step's log if that is not enough.
* **Private repository:** the API needs a token. Without one, I would rely on logs you paste or screenshots.

### Failure triage plan

Fix in this order, one root cause at a time, re-pushing to `master` (or running the workflow manually on a branch) after each fix:

1. `metadata`, `rust (ubuntu)`, `rust (windows)`: should match the local runs; a difference points to the environment (line endings, tool versions).
2. `rust (macos)` and `msrv`: toolchain action inputs, resolver behaviour.
3. `python` rows: wheel tags (manylinux/macOS), old-NumPy wheel availability on Linux/macOS runners, shell path differences.
4. `sdist`: building the extension from the source archive on a runner.
5. Golden-vector failures on Linux or macOS would be a real finding about cross-platform reproducibility; they would be reported and explained, never silently re-recorded.

Numerical algorithms are not to be changed to make CI pass.

## 10. Second validation pass (2026-10-24, commit `2b5e59a`)

The audit fixes (extreme CRR trees, risk scaling, normal-quantile tails, surface
coordinates, Greeks error propagation) are committed as `2b5e59a`. Everything
below ran on Windows 11 x86-64 in a fresh clone of that commit.

| item | result |
|---|---|
| fmt, clippy `-D warnings` (all targets and features), version guard | pass |
| Rust 1.97.1, all features: 171 tests, 93 doctests | pass |
| golden vectors: serial, and `parallel` at 1 and 7 threads | pass; no lock re-recorded |
| Rust 1.74.1, resolver-fallback lockfile, all features: 171 tests, 93 doctests | pass |
| wheel built from the commit, 10 isolated CPython/NumPy pairs, 23 tests each | pass (see `docs/compatibility.md`, including an environment artefact on 3 rows that was rerun) |
| sdist built from the commit, installed by pip in an isolated venv, 23 tests | pass (CPython 3.12.13, NumPy 2.5.4) |
| `cargo audit` on both Rust graphs | no advisories; see `docs/dependency-review.md` |
| `pip-audit` on NumPy | **PYSEC-2021-855 applies to the declared floor 1.21.6**; judged unreachable from quantkit but not tested; floor not changed |
| licence expressions | all permissive or dual-licensed with a permissive option; a third-party notices file for the statically linked dependencies in the wheel does not exist yet |
| package names | **`quantkit` is already taken on PyPI** (project `quantkit` by another author, version 0.0.1rc1). It is free on crates.io. |

### New blocker: the PyPI name

The Python distribution is named `quantkit` in `python/pyproject.toml`. That
name belongs to someone else on PyPI, so it cannot be published as is. The crate
name `quantkit` was free on crates.io when checked (an availability check is not
a reservation; it can change before publishing). A rename of the **distribution**
only (the import name can stay `quantkit`) is a metadata change, but it is a
naming decision, so none was made. Names that returned 404 on both registries at
check time: `quantkit-rs`, `quantkit-py`, `quantkit-options`, `qk-options`,
`optkit-rs`. (`pyquantkit` exists on PyPI.) Importing `quantkit` in a
Python environment that also has the other author's `quantkit` installed would
collide, which is an argument for a different import name too.

### Still not done

* GitHub CI has not run: no repository was provided. `gh` is not installed,
  there is no WSL or Docker on this machine, so Linux and macOS cannot be
  exercised locally either. No wheel for Linux or macOS exists.
* Benchmarks: see the date and conditions at the top of `docs/BENCHMARKS.md`.
