"""Tests that the package only imports what it declares as a dependency."""

import ast
import re
import sys
from pathlib import Path

import pytest

# Python 3.10 has no TOML parser; the check does not depend on the version
tomllib = pytest.importorskip("tomllib")

ROOT = Path(__file__).parents[1]
PACKAGE_DIR = ROOT / "gymnax"
# Imported directly but installed through a declared dependency
TRANSITIVE = {"numpy": "jax"}


def core_dependencies() -> set[str]:
    """Names of the dependencies that a plain `pip install gymnax` installs."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {
        re.split(r"[^A-Za-z0-9_.-]", requirement, maxsplit=1)[0].lower()
        for requirement in project["project"]["dependencies"]
    }


def unconditional_imports(path: Path) -> set[str]:
    """Top-level packages that importing the module at `path` always imports."""
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names - set(sys.stdlib_module_names) - {"gymnax"}


def test_transitive_dependencies_are_backed_by_a_declared_one():
    """Each allowed transitive import names a dependency that is declared."""
    assert set(TRANSITIVE.values()) <= core_dependencies()


def test_modules_import_only_declared_dependencies():
    """Optional packages must be imported lazily or inside `try`, not at the top."""
    allowed = core_dependencies() | set(TRANSITIVE)
    undeclared = {
        f"{path.relative_to(ROOT).as_posix()}: {name}"
        for path in PACKAGE_DIR.rglob("*.py")
        for name in unconditional_imports(path) - allowed
    }
    assert undeclared == set()
