"""The local program of an entity: the engine's effect program or the rule text,
parsed into typed nodes and edges (`Effect`, `effect_tree`, `entity_program`) and
cached by content."""

import json
from dataclasses import dataclass, field
from functools import lru_cache

from main.encoding.fields import Field, fields_of


@dataclass
class Effect:
    """A local program: field rows, relations between them and entity bindings."""

    nodes: list[list[Field]] = field(default_factory=list)
    edges: list[tuple[int, int, str]] = field(default_factory=list)
    bindings: list[tuple[int, str, str]] = field(default_factory=list)
    _tables: dict = field(default_factory=dict, repr=False, compare=False)

    def edge_table(self, vocabulary):
        """Rows of (source, target, relation index), built once per vocabulary."""
        table = self._tables.get(vocabulary.digest)
        if table is None:
            import torch

            encode = vocabulary.relations.encode
            table = torch.tensor(
                [(i, j, encode(role)) for i, j, role in self.edges], dtype=torch.long
            ).reshape(-1, 3)
            self._tables[vocabulary.digest] = table
        return table


def effect_tree(program):
    """Return a bounded cached parse tree; callers treat Effect as immutable."""
    encoded = json.dumps(
        program, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return _effect_tree_cached(encoded)


@lru_cache(maxsize=8192)
def _effect_tree_cached(encoded):
    program = json.loads(encoded)
    effect = Effect()

    def visit(obj, path, parent=None, scope=None):
        scope = dict(scope or {})
        if isinstance(obj, list):
            for i, child in enumerate(obj):
                # Order belongs to the effect program, never to entity packing.
                visit(child, f"{path}[{i}]", parent, scope)
            return
        if not isinstance(obj, dict):
            obj = {"value": obj}
        index = len(effect.nodes)
        own = {
            k: v
            for k, v in obj.items()
            if not isinstance(v, (dict, list))
            and k not in {"ref", "bind"}
            and not (k == "name" and obj.get("kind") == "variable")
        }
        if obj.get("kind") == "variable":
            # Variable names are resolved as program-local identities, not vocab tokens.
            own["variable"] = True
        own["program_role"] = path
        effect.nodes.append(fields_of(own))
        if parent is not None:
            effect.edges.append((parent, index, "program_child"))
            effect.edges.append((index, parent, "program_parent"))
        if obj.get("kind") == "entity" and obj.get("ref"):
            effect.bindings.append((index, obj["ref"], path.split(".")[-1]))
        if obj.get("bind"):
            scope[obj["bind"]] = index
        if obj.get("kind") == "variable" and obj.get("name"):
            binder = scope.get(obj["name"])
            if binder is not None:
                effect.edges.extend(
                    [
                        (binder, index, "binds_variable"),
                        (index, binder, "variable_from"),
                    ]
                )
            else:
                variables.setdefault(obj["name"], []).append(index)
        for key, child in obj.items():
            if key == "refs" and obj.get("kind") == "set":
                for ref in child:
                    effect.bindings.append((index, ref, "set_member"))
                continue
            if isinstance(child, (dict, list)):
                visit(child, key, index, scope)

    variables = {}
    if program:
        visit(program, "root")
    else:
        visit({"kind": "unknown", "known": False}, "root")
    for refs in variables.values():
        for i in refs:
            for j in refs:
                effect.edges.append((i, j, "same_variable"))
    return effect


def entity_program(entity):
    """The engine's program, or the public rule text where the engine has none."""
    from main.encoding import rules

    program = entity.get("semantic_program") or entity.get("program")
    if not program or program.get("op") == "OPAQUE_RULE":
        text = rules.program(entity)
        if text is not None:
            return text
    return effect_tree(program)


def forget():
    """Drop the cached effect trees."""
    _effect_tree_cached.cache_clear()
