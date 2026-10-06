"""Capability classification is not a substitute for native security acceptance."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import _platform_support as support
import pytest


@pytest.mark.parametrize(
    "platform,winerror,should_skip",
    [
        ("nt", 1314, True),
        ("nt", 5, False),
        ("nt", 183, False),
        ("nt", None, False),
        ("posix", 1314, False),
        ("posix", None, False),
    ],
)
def test_symlink_capability_skips_only_windows_privilege_absence(
    tmp_path, monkeypatch, platform, winerror, should_skip
):
    error = OSError("test-owned capability probe")
    if winerror is not None:
        error.winerror = winerror

    def refused(_link, _target, *, target_is_directory=False):
        raise error

    # Only this helper's capability predicate is selected; no OS is emulated.
    monkeypatch.setattr(support, "os", SimpleNamespace(name=platform))
    monkeypatch.setattr(Path, "symlink_to", refused)
    if should_skip:
        with pytest.raises(pytest.skip.Exception, match="WinError 1314"):
            support.symlink_or_skip(tmp_path / "link", tmp_path / "target")
    else:
        with pytest.raises(OSError) as raised:
            support.symlink_or_skip(tmp_path / "link", tmp_path / "target")
        assert raised.value is error


@pytest.mark.parametrize("directory", [False, True])
def test_symlink_capability_creates_real_link_when_available(tmp_path, directory):
    target, link = tmp_path / "target", tmp_path / "link"
    if directory:
        target.mkdir()
        (target / "keep").write_bytes(b"original")
    else:
        target.write_bytes(b"original")
    support.symlink_or_skip(link, target, target_is_directory=directory)
    assert link.is_symlink() and link.resolve() == target.resolve()
    assert (link / "keep" if directory else link).read_bytes() == b"original"
