"""Machine-checkable enforcement of the MPL-2.0 / GPL-3.0 boundary.

The core ``nacre`` distribution is MPL-2.0 and must never import, link, or
acquire a runtime dependency on Gmsh, which is GPL. Until now that rule was
enforced only by discipline, and discipline is what this project deliberately
does not rely on. These tests, plus the ``core-license-boundary`` CI job that
installs the core alone, turn the rule into an invariant that fails loudly.

Two of the three tests run in every job. The third only has meaning where Gmsh
is genuinely absent, so it skips elsewhere.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

CORE_ROOT = Path(__file__).resolve().parents[1] / "nacre" / "src" / "nacre"
CORE_MODULES = (
    "nacre",
    "nacre.contract",
    "nacre.meshir",
    "nacre.checkmesh",
    "nacre.check",
    "nacre.io",
)


def _imported_module_names(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_no_core_module_imports_gmsh() -> None:
    """A static check, so it fails on the offending commit rather than at runtime."""

    sources = sorted(CORE_ROOT.rglob("*.py"))
    assert sources, f"found no core sources under {CORE_ROOT}"
    offenders = {
        str(source.relative_to(CORE_ROOT)): sorted(
            name
            for name in _imported_module_names(source)
            if name == "gmsh" or name.startswith(("gmsh.", "nacre_gmsh"))
        )
        for source in sources
    }
    offenders = {path: names for path, names in offenders.items() if names}
    assert not offenders, (
        f"MPL-2.0 core modules import GPL code: {offenders}. Gmsh integration "
        "belongs in the nacre-gmsh distribution."
    )


def test_importing_the_core_does_not_load_gmsh() -> None:
    """Holds even where Gmsh is installed: the core must not reach for it."""

    program = (
        "import sys\n"
        + "".join(f"import {module}\n" for module in CORE_MODULES)
        + "leaked = [name for name in sys.modules if name.split('.')[0] in "
        "('gmsh', 'nacre_gmsh')]\n"
        "assert not leaked, leaked\n"
        "print('clean')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, (
        f"importing the core pulled in GPL modules:\n{completed.stdout}\n"
        f"{completed.stderr}"
    )


def test_the_core_test_suite_runs_with_gmsh_absent() -> None:
    """The condition the ``core-license-boundary`` CI job creates."""

    if _gmsh_is_installed():
        pytest.skip("gmsh is installed in this environment")
    with pytest.raises(ImportError):
        __import__("gmsh")


def _gmsh_is_installed() -> bool:
    try:
        __import__("gmsh")
    except ImportError:
        return False
    return True


def main() -> int:
    """Standalone gate for the core-only CI job.

    Unlike the tests above this one *demands* that Gmsh be missing, so it is a
    script rather than a test: it is only meaningful in an environment built
    from the core distribution alone.
    """

    import importlib.util

    reachable = [
        name
        for name in ("gmsh", "nacre_gmsh")
        if importlib.util.find_spec(name) is not None
    ]
    if reachable:
        print(
            f"GPL packages are reachable from the core environment: {reachable}",
            file=sys.stderr,
        )
        return 1
    try:
        __import__("gmsh")
    except ImportError as error:
        print(f"import gmsh raises ImportError as required: {error}")
    else:
        print("import gmsh succeeded; the licence boundary is broken", file=sys.stderr)
        return 1

    test_no_core_module_imports_gmsh()
    test_importing_the_core_does_not_load_gmsh()
    print("core licence boundary intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
