"""Fails if the Rust crate, the Python crate and the changelog disagree on the version.

Uses only the standard library and simple pattern matching (no TOML parser), so it
runs on any Python 3.

    python .github/scripts/check_versions.py
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def package_version(path):
    text = (ROOT / path).read_text(encoding="utf-8")
    block = text.split("[package]", 1)[1].split("\n[", 1)[0]
    return re.search(r'^version\s*=\s*"([^"]+)"', block, re.M).group(1)


def python_dependency_version():
    text = (ROOT / "python/Cargo.toml").read_text(encoding="utf-8")
    m = re.search(r'^quantkit\s*=\s*\{[^}]*version\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else None


def main():
    root_version = package_version("Cargo.toml")
    py_version = package_version("python/Cargo.toml")
    dep_version = python_dependency_version()
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## \[([^\]]+)\]", changelog, re.M)
    released = [h for h in headings if h.lower() != "unreleased"]
    pyproject = (ROOT / "python/pyproject.toml").read_text(encoding="utf-8")

    problems = []
    if py_version != root_version:
        problems.append(f"python/Cargo.toml version {py_version} != Cargo.toml version {root_version}")
    if dep_version != root_version:
        problems.append(f"python/Cargo.toml depends on quantkit {dep_version}, expected {root_version}")
    if not released or released[0] != root_version:
        problems.append(f"newest CHANGELOG release heading is {released[:1]}, expected {root_version}")
    if 'dynamic = ["version"]' not in pyproject:
        problems.append("python/pyproject.toml must take its version from Cargo (dynamic = [\"version\"])")

    print(f"Cargo.toml {root_version}; python/Cargo.toml {py_version}; dependency {dep_version}; "
          f"newest changelog release {released[:1]}")
    if problems:
        print("\n".join("ERROR: " + p for p in problems))
        sys.exit(1)
    print("versions are consistent")


if __name__ == "__main__":
    main()
