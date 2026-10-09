"""Bounded Crystal Sphere planning from revealed cells and public geometry only.

Payment and reward collection are policy decisions. Hypothesized boards are a
geometric approximation, not the native RNG posterior. Search utilities only
rank clicks; they are never training rewards or the current-act potential.
"""

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from .protocol import ProtocolError

RULE_VERSION = "crystal-public-geometry-v2"
SIZE = 11

# Public v0.111.0 item shapes, nominal multiplicities and search utilities.
# Different rarities with the same public fragment type share a shape group.
SHAPES = (
    ("CrystalSphereRelic", 4, 4, 1, 100.0),
    ("CrystalSpherePotion", 1, 3, 2, 25.0),
    ("CrystalSpherePotion", 2, 2, 1, 35.0),
    ("CrystalSphereCardReward", 2, 2, 3, 35.0),
    ("CrystalSphereCurse", 2, 2, 1, -120.0),
    ("CrystalSphereGold", 1, 1, 5, 10.0),
    ("CrystalSphereGold", 2, 1, 2, 30.0),
)


def _bit(x, y):
    return 1 << (y * SIZE + x)


@lru_cache(maxsize=None)
def _rectangles(width, height):
    return tuple(sum(_bit(x + dx, y + dy) for dx in range(width) for dy in range(height))
                 for x in range(SIZE - width + 1) for y in range(SIZE - height + 1))


def _refs(candidate, field):
    return candidate.get(field + "s", [candidate.get(field)])


@dataclass(frozen=True)
class Plan:
    candidate: dict
    hypotheses: int
    fallback: bool
    expanded: int

    def diagnostics(self):
        return dict(version=RULE_VERSION, hypotheses=self.hypotheses,
                    geometry_fallback=self.fallback, expanded=self.expanded)


def _public_board(frame):
    entities = frame["public"]["entities"]
    by_ref = {e.get("ref"): e for e in entities if e.get("ref")}
    cells = [e for e in entities if e.get("entity_type") == "board_cell"]
    coordinates = {(e.get("x"), e.get("y")) for e in cells}
    if len(cells) != SIZE * SIZE or coordinates != {(x, y) for x in range(SIZE) for y in range(SIZE)}:
        raise ProtocolError("Crystal Sphere requires the complete public 11x11 board")
    remaining = next((e.get("remaining_steps") for e in entities
                      if e.get("entity_type") == "event_state"), None)
    if type(remaining) is not int or not 1 <= remaining <= 6:
        raise ProtocolError("Crystal Sphere needs a public remaining click count from 1 to 6")
    observed, revealed = defaultdict(int), 0
    kinds = {shape[0] for shape in SHAPES} | {"empty"}
    for cell in cells:
        # Deliberately do not read content_id, item extents or other fields in fog.
        if cell.get("revealed") is not True:
            continue
        kind = cell.get("content_id")
        if kind not in kinds:
            raise ProtocolError("Unknown revealed Crystal Sphere fragment")
        bit = _bit(cell["x"], cell["y"])
        observed[kind] |= bit
        revealed |= bit
    actions = []
    for candidate in frame["legal"]["candidates"]:
        if candidate.get("verb") != "DIVINE_CELL":
            continue
        tools = [by_ref.get(ref, {}).get("content_id") for ref in _refs(candidate, "source_ref")]
        targets = [by_ref.get(ref, {}) for ref in _refs(candidate, "target_ref")]
        tool = next((name for name in tools if name in {"Small", "Big"}), None)
        cell = next((e for e in targets if e.get("entity_type") == "board_cell"), None)
        if tool is None or cell is None:
            raise ProtocolError("Crystal Sphere candidate lacks a public tool or cell")
        x, y = cell["x"], cell["y"]
        center = _bit(x, y)
        if center & revealed:
            raise ProtocolError("Crystal Sphere offered a revealed center")
        radius = int(tool == "Big")
        area = sum(_bit(i, j) for i in range(max(0, x - radius), min(SIZE, x + radius + 1))
                   for j in range(max(0, y - radius), min(SIZE, y + radius + 1)))
        actions.append(((x, y, tool), center, area, candidate))
    if not actions:
        raise ProtocolError("Crystal Sphere has no legal cell action")
    actions.sort(key=lambda action: action[0])
    return remaining, dict(observed), revealed, actions


def _worlds(observed, revealed, rng, count, search_budget):
    placements, by_cell = [], defaultdict(list)
    required = revealed & ~observed.get("empty", 0)
    for index, (kind, width, height, _, _) in enumerate(SHAPES):
        incompatible = revealed & ~observed.get(kind, 0)
        options = [mask for mask in _rectangles(width, height) if not mask & incompatible]
        placements.append(options)
        for mask in options:
            bits = mask & required
            while bits:
                bit = bits & -bits
                by_cell[bit].append((index, mask))
                bits ^= bit
    worlds = []
    expanded = 0
    for _ in range(count):
        budget = search_budget

        def cover(missing, occupied, counts, chosen):
            nonlocal budget, expanded
            if not missing:
                return occupied, counts, chosen
            if budget <= 0:
                return None
            budget -= 1
            expanded += 1
            bits, options = missing, None
            while bits:
                bit = bits & -bits
                valid = [(i, mask) for i, mask in by_cell[bit] if counts[i] and not mask & occupied]
                if not valid:
                    return None
                if options is None or len(valid) < len(options):
                    options = valid
                bits ^= bit
            rng.shuffle(options)
            for index, mask in options:
                rest = list(counts)
                rest[index] -= 1
                found = cover(missing & ~mask, occupied | mask, rest, chosen + [(index, mask)])
                if found is not None:
                    return found
            return None

        found = cover(required, 0, [shape[3] for shape in SHAPES], [])
        if found is None:
            continue
        occupied, counts, chosen = found
        for index, count_left in enumerate(counts):
            for _ in range(count_left):
                options = [mask for mask in placements[index] if not mask & occupied]
                if not options:
                    break  # Native item placement can fail; do not invent occupied cells.
                mask = rng.choice(options)
                occupied |= mask
                chosen.append((index, mask))
        worlds.append(chosen)
    weights = defaultdict(float)
    if worlds:
        for world in worlds:
            for index, mask in world:
                weights[mask] += SHAPES[index][4] / len(worlds)
    else:
        # A bounded constraint search can fail. Keep a public geometric fallback
        # rather than read hidden extents or fail the run. Its use is reported.
        for shape, options in zip(SHAPES, placements):
            for mask in options:
                weights[mask] += shape[3] * shape[4] / len(options)
    return tuple(weights.items()), len(worlds), expanded


def plan(frame, *, samples=12, beam_width=8, search_budget=512):
    if frame["public"]["phase"] != "crystal_sphere":
        return None
    if any(type(value) is not int or value < 1 for value in (samples, beam_width, search_budget)):
        raise ValueError("Crystal planning budgets must be positive integers")
    remaining, observed, revealed, actions = _public_board(frame)
    # Reproducible across Steam/headless and independent of the game seed,
    # routing IDs, hidden values, and entity/candidate packing order.
    key = json.dumps([RULE_VERSION, remaining, sorted(observed.items())]).encode()
    rng = random.Random(hashlib.sha256(key).digest())
    weights, worlds, expanded = _worlds(observed, revealed, rng, samples, search_budget)

    def score(mask, steps_left):
        value = 0.0
        for item, weight in weights:
            opened = (item & mask).bit_count()
            size = item.bit_count()
            if opened == size:
                value += weight
            elif steps_left and weight > 0 and size - opened <= steps_left * 9:
                value += 0.2 * weight * (opened / size) ** 3
        return value

    beam = [(revealed, None)]
    for depth in range(remaining):
        candidates = {}
        for mask, first in beam:
            for index, (_, center, area, _) in enumerate(actions):
                if center & mask:
                    continue  # Later actions must still have an unrevealed center.
                after = mask | area
                initial = index if first is None else first
                if after not in candidates:
                    candidates[after] = initial
        if not candidates:
            break
        expanded += len(candidates)
        ranked = sorted(candidates, key=lambda mask: (
            score(mask, remaining - depth - 1), mask.bit_count(), -candidates[mask]), reverse=True)
        beam = [(mask, candidates[mask]) for mask in ranked[:beam_width]]
    first = beam[0][1]
    if first is None:
        raise ProtocolError("Crystal Sphere planner found no executable action")
    return Plan(actions[first][3], worlds, worlds == 0, expanded)


def collection_candidate(frame):
    result = plan(frame)
    return result.candidate if result is not None else None
