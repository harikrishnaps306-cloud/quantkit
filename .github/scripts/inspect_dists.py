#!/usr/bin/env python3
"""Inspect the built distributions in a directory and fail if they are wrong.

    python .github/scripts/inspect_dists.py DIST_DIR [--expect-platforms linux,macos,windows]

For every wheel it prints the filename, the tags from the WHEEL file, the
metadata that matters to installers and the licence files, and it asserts:

* the tags are the expected abi3 (CPython 3.9+) tags, and the platform tags are
  manylinux x86-64, macOS arm64 and Windows x86-64 for the platforms requested;
* `Requires-Dist` contains exactly the NumPy requirement declared in
  python/pyproject.toml, that the requirement excludes 1.21.6 (the last release
  before the fixed 1.22) and admits 1.22.0, and that no other runtime
  dependency crept in;
* the project name, version and Requires-Python match;
* LICENSE-MIT.txt, LICENSE-APACHE.txt and a non-empty THIRD_PARTY_NOTICES.txt are
  in the wheel's dist-info/licenses and listed as License-File;
* the sdist has the same name, version and requirement and contains the notices
  and no build outputs.

It reads files only; it needs `packaging` for the requirement check.
"""
import re
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]
NAME, VERSION = "quantkit-rs", re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "Cargo.toml").read_text(encoding="utf-8"), re.M).group(1)
EXPECT_REQ = re.search(r'dependencies\s*=\s*\["([^"]+)"\]', (ROOT / "python/pyproject.toml").read_text(encoding="utf-8")).group(1)
PLATFORM_RE = {
    "linux": re.compile(r"manylinux_\d+_\d+_x86_64"),
    "macos": re.compile(r"macosx_\d+_\d+_arm64"),
    "windows": re.compile(r"win_amd64"),
}
failures = []


def check(cond, message):
    print(("  ok   " if cond else "  FAIL ") + message)
    if not cond:
        failures.append(message)


def check_metadata(meta, where):
    check(meta["Name"] == NAME, f"{where}: Name is {meta['Name']!r}, expected {NAME!r}")
    check(meta["Version"] == VERSION, f"{where}: Version is {meta['Version']}, expected {VERSION}")
    check(meta["Requires-Python"] == ">=3.9", f"{where}: Requires-Python is {meta['Requires-Python']!r}")
    reqs = meta.get_all("Requires-Dist") or []
    check(reqs == [EXPECT_REQ], f"{where}: Requires-Dist is {reqs}, expected [{EXPECT_REQ!r}]")
    if reqs:
        spec = Requirement(reqs[0]).specifier
        check(not spec.contains(Version("1.21.6"), prereleases=True), f"{where}: the NumPy requirement excludes 1.21.6 (affected by GHSA-fpfv-jqm9-f5jm)")
        check(spec.contains(Version("1.22.0")), f"{where}: the NumPy requirement admits 1.22.0, the oldest tested release")
        check(spec.contains(Version("2.5.4")), f"{where}: the NumPy requirement admits 2.5.4, the newest tested release")
    check(meta["License-Expression"] == "MIT OR Apache-2.0", f"{where}: License-Expression is {meta['License-Expression']!r}")
    files = meta.get_all("License-File") or []
    for f in ("LICENSE-MIT.txt", "LICENSE-APACHE.txt", "THIRD_PARTY_NOTICES.txt"):
        check(f in files, f"{where}: License-File lists {f}")
    urls = " ".join(meta.get_all("Project-URL") or [])
    check("github.com/harikrishnaps306-cloud/quantkit" in urls, f"{where}: Project-URL points at the repository")


def inspect_wheel(path, wanted):
    print(f"\n{path.name}")
    stem = path.name[: -len(".whl")].split("-")
    tags = "-".join(stem[2:])
    print(f"  tags: {tags}")
    check(stem[0] == "quantkit_rs" and stem[1] == VERSION, "filename carries the distribution name and version")
    check(stem[2] == "cp39" and stem[3] == "abi3", "interpreter and ABI tags are cp39-abi3 (CPython 3.9 and later)")
    platform_ok = [p for p, rx in PLATFORM_RE.items() if rx.search(stem[4])]
    print(f"  platform tag: {stem[4]} -> {platform_ok}")
    check(len(platform_ok) == 1 and platform_ok[0] in wanted, f"platform tag is one of the expected platforms {sorted(wanted)}")
    z = zipfile.ZipFile(path)
    names = z.namelist()
    di = next(n.split("/")[0] for n in names if n.endswith(".dist-info/METADATA"))
    meta = Parser().parsestr(z.read(f"{di}/METADATA").decode("utf-8"))
    wheel = Parser().parsestr(z.read(f"{di}/WHEEL").decode("utf-8"))
    print(f"  WHEEL tags: {wheel.get_all('Tag')}  Root-Is-Purelib: {wheel['Root-Is-Purelib']}")
    check(wheel["Root-Is-Purelib"] == "false", "wheel is marked platform-specific (not purelib)")
    check_metadata(meta, path.name)
    for lic in ("LICENSE-MIT.txt", "LICENSE-APACHE.txt", "THIRD_PARTY_NOTICES.txt"):
        check(f"{di}/licenses/{lic}" in names, f"{lic} is inside dist-info/licenses")
    notices = z.read(f"{di}/licenses/THIRD_PARTY_NOTICES.txt").decode("utf-8")
    check(len(notices) > 100_000 and "Packages listed:" in notices, f"notices file is substantial ({len(notices)} bytes)")
    exts = [n for n in names if re.search(r"_quantkit\.(pyd|so|abi3\.so|cpython[^/]*\.so)$", n)]
    check(len(exts) == 1, f"exactly one compiled extension: {exts}")
    check("quantkit/py.typed" in names and "quantkit/__init__.py" in names, "package files present (__init__.py, py.typed)")
    return platform_ok[0] if platform_ok else None


def inspect_sdist(path):
    print(f"\n{path.name}")
    t = tarfile.open(path)
    names = t.getnames()
    top = names[0].split("/")[0]
    meta = Parser().parsestr(t.extractfile(f"{top}/PKG-INFO").read().decode("utf-8"))
    check_metadata(meta, path.name)
    check(f"{top}/THIRD_PARTY_NOTICES.txt" in names, "sdist contains THIRD_PARTY_NOTICES.txt")
    bad = [n for n in names if re.search(r"(^|/)(target|dist)/|\.(pyd|whl|so)$", n)]
    check(not bad, f"sdist has no build outputs {bad[:3]}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dist = Path(args[0])
    wanted = {"linux", "macos", "windows"}
    for a in sys.argv[1:]:
        if a.startswith("--expect-platforms="):
            wanted = set(a.split("=", 1)[1].split(","))
    print(f"expecting {NAME} {VERSION}, requirement {EXPECT_REQ!r}, platforms {sorted(wanted)}")
    wheels = sorted(dist.glob("*.whl"))
    seen = [inspect_wheel(w, wanted) for w in wheels]
    check(sorted(p for p in seen if p) == sorted(wanted), f"exactly one wheel per expected platform (found {seen})")
    sdists = list(dist.glob("*.tar.gz"))
    check(len(sdists) == 1, f"exactly one sdist (found {len(sdists)})")
    for s in sdists:
        inspect_sdist(s)
    print()
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all distribution checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
