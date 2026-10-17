"""The act map compressed to the current node and the map frontier; each frontier
node carries the map reachable from it as a local program (`compress_map`,
`map_relations`)."""

import json
from functools import lru_cache

from main.env.protocol import ProtocolError
from main.encoding.fields import fields_of
from main.encoding.programs import Effect


MAP_DYNAMIC_FIELDS = {"ref", "current", "visited", "selectable"}


def compress_map(entities, effects, relations, selected):
    """Keep the current map node and its frontier; fold the rest into programs.

    The frontier is every successor of the current node (the entry nodes when
    no node is current) plus any map node another entity still refers to. Each
    frontier node receives the map reachable from it as its local program, so
    nodes which can no longer be reached, and the path already taken, leave
    the observation. While the player may ignore paths (``free_travel``), every
    node of the next floor is a successor.
    """
    nodes = {
        x["ref"]: x
        for x in entities
        if x.get("entity_type") == "map_node" and x.get("ref")
    }
    if not nodes:
        return entities, effects, relations, {}
    forward = {ref: [] for ref in nodes}
    entered = set()
    referenced = set(selected)
    for edge in relations:
        source, target = edge.get("source"), edge.get("target")
        if edge.get("role") == "map_edge" and source in nodes and target in nodes:
            forward[source].append(target)
            entered.add(target)
        else:
            referenced.update((source, target))
    for item, effect in zip(entities, effects):
        if item.get("entity_type") == "map_node":
            continue
        for role in ("source", "target", "option", "owner"):
            referenced.update(item.get(role + "_refs", []))
            referenced.add(item.get(role + "_ref"))
        referenced.update(ref for _, ref, _ in effect.bindings)
    free = any(x.get("free_travel") is True for x in entities)
    if free:
        floors = {}
        for ref, x in nodes.items():
            floors.setdefault(x.get("floor"), []).append(ref)
        steps = {
            ref: sorted(set(forward[ref]) | set(floors.get(x["floor"] + 1, ())))
            if type(x.get("floor")) is int
            else forward[ref]
            for ref, x in nodes.items()
        }
    else:
        steps = forward
    current = [ref for ref, x in nodes.items() if x.get("current") is True]
    frontier = (
        {target for ref in current for target in steps[ref]}
        if current
        else set(nodes) - entered
    )
    keep = set(current) | frontier | (referenced & nodes.keys())
    keys = {
        ref: _map_key(ref, nodes, forward, steps)
        for ref in sorted(keep - set(current))
    }
    kept_entities, kept_effects = [], []
    for item, effect in zip(entities, effects):
        if item.get("entity_type") == "map_node":
            ref = item.get("ref")
            if ref not in keep:
                continue
            if ref in keys:
                effect = _map_program(keys[ref])
        kept_entities.append(item)
        kept_effects.append(effect)
    kept_relations = [
        edge
        for edge in relations
        if not (
            edge.get("role") == "map_edge"
            and edge.get("source") in nodes
            and edge.get("target") in nodes
            and not (edge["source"] in keep and edge["target"] in keep)
        )
    ]
    return kept_entities, kept_effects, kept_relations, keys


def _map_key(root, nodes, forward, steps):
    """Canonical description of the map reachable from root; root is node 0.

    ``forward`` holds the drawn paths and ``steps`` the moves actually allowed,
    which also include every next-floor node during free travel.
    """
    reachable, stack = {root}, [root]
    while stack:
        for target in steps[stack.pop()]:
            if target not in reachable:
                reachable.add(target)
                stack.append(target)
    order = [root] + sorted(reachable - {root})
    index = {ref: i for i, ref in enumerate(order)}
    static = [
        {k: v for k, v in nodes[ref].items() if k not in MAP_DYNAMIC_FIELDS}
        for ref in order
    ]
    links = sorted(
        (index[ref], index[target])
        for ref in order
        for target in forward[ref]
        if target in index
    )
    free = sorted(
        (index[ref], index[target])
        for ref in order
        for target in steps[ref]
        if target not in forward[ref]
    )
    return json.dumps(
        [static, links, free], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


@lru_cache(maxsize=4096)
def _map_program(encoded):
    """The reachable map as a program: a map is fixed for an act, so this is cached."""
    static, links, free = json.loads(encoded)
    effect = Effect()
    base = static[0].get("floor")
    for node in static:
        floor = node.get("floor")
        if all(type(x) in (int, float) for x in (base, floor)):
            # Distance ahead of the frontier node, in floors.
            node = dict(node, depth=floor - base)
        effect.nodes.append(fields_of(node))
    effect.edges.extend(
        map_relations(len(static), [tuple(x) for x in links], [tuple(x) for x in free])
    )
    return effect


def map_relations(count, links, free=()):
    """Classify every ordered node pair of a map by reachability along its paths.

    ``free`` pairs are single moves which no path provides; they are labelled
    as such instead of unreachable.
    """
    direct, free = set(links), set(free)
    adjacency = {i: [j for a, j in links if a == i] for i in range(count)}
    reachable = {}

    def reach(i, stack):
        if i in stack:
            raise ProtocolError("Map must be a DAG")
        if i not in reachable:
            reachable[i] = set()
            for j in adjacency[i]:
                reachable[i].add(j)
                reachable[i].update(reach(j, stack | {i}))
        return reachable[i]

    for i in range(count):
        reach(i, set())
    result = []
    for i in range(count):
        for j in range(count):
            role = (
                "self"
                if i == j
                else "forward_direct"
                if (i, j) in direct
                else "reverse_direct"
                if (j, i) in direct
                else "forward_indirect"
                if j in reachable[i]
                else "reverse_indirect"
                if i in reachable[j]
                else "free_direct"
                if (i, j) in free
                else "free_reverse"
                if (j, i) in free
                else "unreachable"
            )
            result.append((i, j, "map_" + role))
    return result


def forget():
    """Drop the cached map programs."""
    _map_program.cache_clear()
