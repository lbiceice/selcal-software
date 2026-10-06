"""Capability checks for tests that need symlinks or FIFOs (Windows test report, 2026-09-24).

Creating a symlink on Windows needs Developer Mode or administrator rights (WinError 1314), and
Windows has no ``os.mkfifo``. Tests that exercise these protections skip with that reason where
the capability is missing; on POSIX they always run. A skip is reported, never counted as a pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

WINDOWS_SYMLINK_REASON = (
    "creating symlinks needs Developer Mode or administrator rights on Windows (WinError 1314); "
    "run the security test lane with that capability"
)

requires_fifo = pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="os.mkfifo is unavailable on this platform (Windows)"
)


def symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    """Create ``link`` -> ``target``; skip the test only if Windows lacks the privilege."""
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip(WINDOWS_SYMLINK_REASON)
        raise


def directory_link_or_skip(link: Path, target: Path) -> None:
    """Create a directory link ``link`` -> ``target`` for product link-refusal tests.

    Uses a symlink where permitted. Without the Windows symlink privilege it creates a
    directory junction instead, which needs no privilege and is also a name-surrogate reparse
    point, so the product's directory-link protections are exercised on ordinary Windows
    accounts (R10v2 Windows return, 2026-10-03: 66 link tests had been skipped). The test
    skips only if neither link can be created. Use it only where the assertions depend on
    the product's behaviour, not on symlink-specific details such as ``readlink``.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError as error:
        if not (os.name == "nt" and getattr(error, "winerror", None) == 1314):
            raise
    try:
        import _winapi

        _winapi.CreateJunction(os.path.abspath(target), os.path.abspath(link))
    except (ImportError, AttributeError, OSError) as error:
        pytest.skip(f"{WINDOWS_SYMLINK_REASON}; directory junction also unavailable: {error}")


def mkfifo_or_skip(path: Path, mode: int = 0o666) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("os.mkfifo is unavailable on this platform (Windows)")
    os.mkfifo(path, mode)


posix_file_modes_only = pytest.mark.skipif(
    os.name == "nt",
    reason=(
        "checks the frozen POSIX file-mode contract (0644) of a historical internal review "
        "package; Windows has no POSIX permission bits (chmod only sets read-only)"
    ),
)


def _selcal_is_the_source_checkout() -> bool:
    import selcal

    return (
        Path(selcal.__file__).resolve()
        == Path(__file__).resolve().parents[1] / "src" / "selcal" / "__init__.py"
    )


requires_source_checkout = pytest.mark.skipif(
    not _selcal_is_the_source_checkout(),
    reason=(
        "development diagnostics: the harness refuses to run unless SelCal is imported from this "
        "worktree's src/ (it binds results to the checked-out source), so it cannot run against "
        "an installed wheel or sdist"
    ),
)
