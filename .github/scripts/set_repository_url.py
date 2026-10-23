"""Sets the verified repository URL in the package metadata, once a repository exists.

Dry run by default (prints what would change); pass --write to edit files.

    python .github/scripts/set_repository_url.py https://github.com/OWNER/NAME
    python .github/scripts/set_repository_url.py https://github.com/OWNER/NAME --write

It only ever uses the URL you give it. It refuses anything that is not a plain
`https://github.com/OWNER/NAME` URL, and it does not talk to the network: confirm
that the repository really exists (for example `git ls-remote <url>`) before
running it with --write.

Files edited:
  Cargo.toml             adds `repository = "<url>"` (and `homepage` is left unset)
  python/pyproject.toml  adds a [project.urls] table (Repository, Issues, Changelog)
  README.md              adds a CI status badge under the title
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
URL_RE = re.compile(r"^https://github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)/([A-Za-z0-9._-]+)$")


def plan(url, branch):
    changes = {}

    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    if re.search(r"^repository\s*=", cargo, re.M):
        raise SystemExit("Cargo.toml already has a repository field; edit it by hand")
    new = re.sub(r"^(license\s*=.*\n)", rf'\1repository = "{url}"\n', cargo, count=1, flags=re.M)
    if new == cargo:
        raise SystemExit("could not find the license line in Cargo.toml to anchor the edit")
    changes["Cargo.toml"] = new

    pyproject = (ROOT / "python/pyproject.toml").read_text(encoding="utf-8")
    if "[project.urls]" in pyproject:
        raise SystemExit("python/pyproject.toml already has [project.urls]; edit it by hand")
    urls = (
        "\n[project.urls]\n"
        f'Repository = "{url}"\n'
        f'Issues = "{url}/issues"\n'
        f'Changelog = "{url}/blob/{branch}/CHANGELOG.md"\n'
    )
    new = pyproject.replace("\n[tool.maturin]", urls + "\n[tool.maturin]", 1)
    if new == pyproject:
        raise SystemExit("could not find [tool.maturin] in python/pyproject.toml to anchor the edit")
    changes["python/pyproject.toml"] = new

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    badge = f"[![CI]({url}/actions/workflows/ci.yml/badge.svg?branch={branch})]({url}/actions/workflows/ci.yml)"
    if badge not in readme:
        new = readme.replace("# quantkit\n", f"# quantkit\n\n{badge}\n", 1)
        if new == readme:
            raise SystemExit("could not find the README title to anchor the badge")
        changes["README.md"] = new
    return changes


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    write = "--write" in argv
    if len(args) != 1:
        raise SystemExit(__doc__)
    url = args[0].rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    if not URL_RE.match(url):
        raise SystemExit(f"not a plain https://github.com/OWNER/NAME URL: {url!r}")
    branch = "master"
    changes = plan(url, branch)
    for name, text in changes.items():
        old = (ROOT / name).read_text(encoding="utf-8")
        added = [l for l in text.splitlines() if l not in old.splitlines()]
        print(f"{'WRITE' if write else 'would change'} {name}:")
        for line in added:
            print("   +", line)
        if write:
            (ROOT / name).write_text(text, encoding="utf-8")
    if not write:
        print("\n(dry run; nothing was written. Re-run with --write after confirming the repository exists.)")


if __name__ == "__main__":
    main(sys.argv[1:])
