"""Make shared helpers in tests/ importable from subdirectories such as tests/task10/."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS = str(Path(__file__).resolve().parent)
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)


def pytest_sessionstart(session: object) -> None:
    """Refuse a process in which SIGINT is ignored (POSIX only).

    A job started with ``&`` from a non-interactive shell inherits SIGINT ignored, and so does
    every child it starts. The real-helper tests cancel a child with SIGINT and only kill it after
    two seconds, so in such a process they fail late with a misleading message instead of here.
    """
    import os
    import signal

    if os.name != "nt" and signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        raise pytest.UsageError(
            "SIGINT is ignored in this process (for example a background job of a non-interactive"
            " shell); the real-helper cancel tests need the default disposition. Run pytest in the"
            " foreground, or from a launcher that does not ignore SIGINT."
        )
