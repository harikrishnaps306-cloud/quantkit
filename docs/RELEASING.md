# Releasing quantkit

Nothing in the repository publishes automatically. The `Release` workflow
(`.github/workflows/release.yml`) is manual-only, and crates.io is published by
hand. Both registries are permanent: a published version cannot be replaced or
deleted (it can only be yanked), and a name once used cannot be reassigned by
the publisher. Read this page before the first release.

## One-time setup (done by the maintainer, not by CI)

**PyPI (trusted publishing, no token stored in GitHub)**

1. Create or sign in to a PyPI account with two-factor authentication.
2. At <https://pypi.org/manage/account/publishing/> add a *pending publisher*:
   project name `quantkit-rs`, owner `harikrishnaps306-cloud`, repository
   `quantkit`, workflow file `release.yml`, environment `pypi`.
3. In the GitHub repository, Settings, Environments: create an environment named
   `pypi` and add **required reviewers** (yourself). That approval is the last
   gate before an upload.

**crates.io**

1. Sign in at <https://crates.io> with the GitHub account, verify the email, and
   create an API token with only the publish scope at
   <https://crates.io/settings/tokens>.
2. Run `cargo login` in your own terminal and paste the token there. Do not paste
   it into a chat, an issue or a workflow file.

## Release steps

1. Final commit on `master` with CI green (check the Actions tab; the badge in the
   README tracks `master`). `CHANGELOG.md` must have a heading for the version
   (the `0.x` rules are in `docs/VERSIONING.md`).
2. Re-check the names immediately beforehand (HTTP 404 means free):

   ```bash
   curl -s -o /dev/null -w '%{http_code}\n' -A 'name-check' https://crates.io/api/v1/crates/quantkit
   curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/quantkit-rs/json
   ```

3. **Dry run of the Python release.** Actions, Release, Run workflow, branch
   `master`, `publish` unchecked. It builds the three wheels and the sdist,
   installs each wheel in a clean virtualenv and runs the tests, and runs
   `twine check --strict`. Download the `wheel-*` and `sdist` artifacts if you
   want to inspect them.
4. **Publish to PyPI.** Run the same workflow with `publish` checked and
   `confirm_version` set to the exact version (for example `0.2.0`). It refuses
   unless the branch is `master` and the confirmation matches. Approve the `pypi`
   environment prompt when it appears.
5. **Publish to crates.io** from a clean checkout of the same commit:

   ```bash
   cargo publish --dry-run
   cargo publish
   ```

6. **Tag and release notes**, after both uploads succeed (the workflow does not do
   this): `git tag v0.2.0 && git push origin v0.2.0`, then create a GitHub release
   from `docs/RELEASE_NOTES_0.2.0.md`. Tag pushes do not trigger CI.
7. **Check what users get:** in a fresh virtualenv, `pip install quantkit-rs`,
   `python -c "import quantkit; print(quantkit.bs_price(42, 40, 0.10, 0.20, 0.5))"`
   (about 4.759422); and `cargo add quantkit` in a scratch crate.

## What the workflow builds and does not build

Wheels for Linux x86-64 (manylinux), macOS arm64 and Windows x86-64, one abi3
wheel each (CPython 3.9 and later), plus the sdist. There are no wheels for macOS
Intel, Linux ARM, musl or 32-bit systems; those users build from the sdist, which
needs a Rust toolchain, and they are unsupported (`docs/VERSIONING.md`).

The workflow's own steps (action tags, manylinux container, trusted-publishing
exchange) are only proven by running it. A dry run exercises everything except the
upload to PyPI; the upload itself can only be proven by a real release.

## If something goes wrong

* A broken release: yank it (PyPI: project, Manage, the release, Yank; crates.io:
  `cargo yank --version 0.2.0 quantkit`), fix, and release the next patch version.
  Yanking hides a version from new resolutions; it does not delete it.
* A leaked token: revoke it on the registry first, then investigate.
