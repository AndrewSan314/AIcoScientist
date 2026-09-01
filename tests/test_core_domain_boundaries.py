from __future__ import annotations

import ast
from pathlib import Path
import pytest


def _get_imports_from_file(file_path: Path) -> set[str]:
    """Parses a Python file AST and returns all imported module paths."""
    content = file_path.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(file_path))
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def test_core_does_not_import_auirh_domain() -> None:
    """Verifies that generic core science abstractions do not statically import domain packs."""
    core_modules = [
        Path("src/science/actions.py"),
        Path("src/science/domain.py"),
        Path("src/science/records.py"),
        Path("src/science/hypothesis_models.py"),
        Path("src/science/falsification/information_gain.py"),
        Path("src/science/falsification/policy.py"),
        Path("src/science/provenance.py"),
        Path("src/optimization/backend.py"),
        Path("src/optimization/objective.py"),
    ]

    forbidden_roots = (
        "src.domains.auirh",
        "src.domains.toy_material",
        "src.domains",
        "src.datasets.auirh_actions",
        "src.datasets.auirh",
    )

    for mod_path in core_modules:
        assert mod_path.is_file(), f"Expected core file {mod_path} to exist."
        imports = _get_imports_from_file(mod_path)
        for imp in imports:
            for forbidden in forbidden_roots:
                assert not imp.startswith(forbidden), (
                    f"Architecture Boundary Violation: Core module '{mod_path}' statically imports '{imp}'."
                )
