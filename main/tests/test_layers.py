"""The subpackages of `main` depend on each other in one direction only.

    env <- encoding <- data, network <- play <- learn <- iterate <- cli
                                  demos <- steam <- cli

A subpackage imports from those of lower rank and from itself; `data` and `network`
stand side by side and import neither each other. The modules at the top (`iterate`,
`steam`) may import anything below them; `cli` is the entry and imports them,
and nothing imports `cli`.
"""

import ast
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1]
RANK = dict(env=0, encoding=1, data=2, network=2, play=3, demos=3, learn=4)
TOP = 9


def unit(module):
    """The subpackage a dotted module of `main` belongs to; None outside `main`."""
    parts = module.split(".")
    if parts[0] != "main":
        return None
    return parts[1] if len(parts) > 1 and parts[1] in RANK else "top"


def imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module
        elif isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)


def test_no_subpackage_imports_one_above_or_beside_it():
    wrong = []
    for path in MAIN.rglob("*.py"):
        if "tests" in path.parts:
            continue
        module = ".".join(path.relative_to(MAIN.parent).with_suffix("").parts)
        own = unit(module)
        rank = RANK.get(own, TOP)
        for target in imports(path):
            other = unit(target)
            if other is None or other == own or own == "top":
                continue
            if RANK.get(other, TOP) >= rank:
                wrong.append(f"{module} -> {target}")
    assert not wrong, "\n".join(sorted(wrong))


def test_only_the_top_reaches_outside_main():
    """Below the top, `main` does not import the demonstration and solver pipelines,
    which import it."""
    wrong = []
    for path in MAIN.rglob("*.py"):
        if "tests" in path.parts:
            continue
        module = ".".join(path.relative_to(MAIN.parent).with_suffix("").parts)
        if unit(module) == "top":
            continue
        wrong += [f"{module} -> {target}" for target in imports(path)
                  if target.split(".")[0] in ("spire_codex_data", "combat_solver_cli")]
    assert not wrong, "\n".join(sorted(wrong))


def test_nothing_imports_the_command_line_entry():
    wrong = []
    for path in MAIN.rglob("*.py"):
        if "tests" in path.parts or path.name == "__main__.py":
            continue
        module = ".".join(path.relative_to(MAIN.parent).with_suffix("").parts)
        if module == "main.cli":
            continue
        wrong += [f"{module} -> {target}" for target in imports(path)
                  if target == "main.cli" or target.startswith("main.cli.")]
    assert not wrong, "\n".join(sorted(wrong))
