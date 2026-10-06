"""The shipped user documentation, read as one text.

The README is the entry page; the complete user reference and the development and platform
record moved to ``docs/guide/`` on 2026-10-05. Assertions about what the documentation states
read all three pages, so moving a paragraph between them does not weaken any assertion.
"""

from __future__ import annotations

from pathlib import Path

PUBLIC_DOCUMENTATION_PAGES = (
    "README.md",
    "docs/guide/reference.md",
    "docs/guide/development.md",
)


def public_documentation_text(root: Path) -> str:
    return "\n".join(
        (root / page).read_text(encoding="utf-8") for page in PUBLIC_DOCUMENTATION_PAGES
    )
