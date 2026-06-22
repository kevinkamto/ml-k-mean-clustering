"""Project linting and type-checking entry point.

Runs, in order:

1. ``ruff check --fix``  - lint and auto-fix
2. ``ruff format``       - format code
3. ``mypy``              - static type checking

Usage::

    uv run python lint.py          # fix + format + type-check
    uv run python lint.py --check  # no changes; fail if not clean (for CI)
"""

from __future__ import annotations

import subprocess
import sys

# Paths that the tools operate on.
TARGETS = ["src", "tests", "lint.py"]


def _run(cmd: list[str]) -> int:
    """Run a command, echoing it, and return its exit code."""
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    check_only = "--check" in argv

    if check_only:
        steps = [
            ["ruff", "check", *TARGETS],
            ["ruff", "format", "--check", *TARGETS],
            ["mypy"],
        ]
    else:
        steps = [
            ["ruff", "check", "--fix", *TARGETS],
            ["ruff", "format", *TARGETS],
            ["mypy"],
        ]

    failed = 0
    for step in steps:
        if _run(step) != 0:
            failed += 1

    if failed:
        print(f"\n{failed} step(s) failed.")
        return 1
    print("\nAll lint steps passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
