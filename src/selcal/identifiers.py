"""Shared validation for versioned scientific-contract identifiers."""

from __future__ import annotations

import re

_VERSIONED_NAME = re.compile(r"[a-z][a-z0-9_]*_v[1-9][0-9]*\Z")


def require_versioned_name(value: object, *, name: str) -> str:
    """Require an exact built-in, lower-case versioned contract name."""

    if type(value) is not str:
        raise ValueError(f"{name} must be a built-in string")
    if _VERSIONED_NAME.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lower-case versioned name")
    return value
