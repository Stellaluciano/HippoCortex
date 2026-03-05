#!/usr/bin/env python3
"""Fail if a Python package directory contains a nested mirror with the same name."""

from __future__ import annotations

from pathlib import Path


def is_python_package(path: Path) -> bool:
    return path.is_dir() and (path / "__init__.py").exists()


def find_duplicate_mirrors(root: Path) -> list[Path]:
    mirrors: list[Path] = []
    for init_file in root.rglob("__init__.py"):
        pkg_dir = init_file.parent
        nested_same = pkg_dir / pkg_dir.name
        if is_python_package(nested_same):
            mirrors.append(nested_same)
    return sorted(set(mirrors))


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    mirrors = find_duplicate_mirrors(repo_root)
    if not mirrors:
        print("No duplicate package mirror directories found.")
        return 0

    print("Found duplicate package mirror directories:")
    for mirror in mirrors:
        print(f" - {mirror.relative_to(repo_root)}")
    print("Please keep a single authoritative package source directory.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
