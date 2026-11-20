"""Tests that the documentation under `docs/` covers the public API."""

import re
from pathlib import Path

import pytest

import gymnax
from gymnax import registration, wrappers

DOCS_DIR = Path(__file__).parents[2] / "docs"
DIRECTIVE = re.compile(r"^::: (\S+)$", re.MULTILINE)


@pytest.fixture(scope="module")
def documented() -> set[str]:
    """Identifiers of all `:::` directives in the API reference."""
    return {
        identifier
        for page in (DOCS_DIR / "reference").rglob("*.md")
        for identifier in DIRECTIVE.findall(page.read_text())
    }


def qualified_name(obj) -> str:
    """Return the dotted path that a `:::` directive uses for an object."""
    return f"{obj.__module__}.{obj.__qualname__}"


def test_every_builtin_environment_has_a_reference_page(documented):
    """Adding a built-in environment requires adding a reference page."""
    missing = [
        env_id
        for env_id, env_class in registration._builtins.items()
        if qualified_name(env_class) not in documented
    ]
    assert missing == []


def test_every_builtin_environment_is_listed_in_the_overview():
    """Adding a built-in environment requires adding a row to the overview."""
    overview = (DOCS_DIR / "environments.md").read_text()
    missing = [
        env_id for env_id in registration._builtins if f"`{env_id}`" not in overview
    ]
    assert missing == []


def test_every_exported_wrapper_is_documented(documented):
    """Wrappers exported by `gymnax.wrappers` are part of the API reference."""
    missing = [
        name
        for name in wrappers.__all__
        if qualified_name(getattr(wrappers, name)) not in documented
    ]
    assert missing == []


def test_public_entry_points_are_documented(documented):
    """Functions exported from the top-level package are documented."""
    missing = [
        name
        for name in ("make", "register")
        if qualified_name(getattr(gymnax, name)) not in documented
    ]
    assert missing == []
