"""Shared native replay and public action priors for MCTS."""

import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from model.protocol import (
    action_semantics,
    execution_command,
    fingerprint,
    validate_frame,
)

from .client import SolverEngine


class SearchLimit(RuntimeError):
    pass


class ReplayMismatch(RuntimeError):
    pass


def state_key(frame):
    # Verification only. Equal public observations do not imply equal hidden RNG.
    return fingerprint(
        {
            "public": frame["public"],
            "candidates": [
                action_semantics(c)
                for c in frame.get("legal", {}).get("candidates", [])
            ],
        }
    )


def resolve(frame, semantics):
    matches = [
        c for c in frame["legal"]["candidates"] if action_semantics(c) == semantics
    ]
    if len(matches) != 1:
        raise ReplayMismatch("Recorded action has no unique native candidate")
    return matches[0]


@dataclass(eq=False, frozen=True)
class Prefix:
    parent: object
    before_hash: str
    action: dict
    actor: str
    length: int
    phase: str = ""
    act: int = 0
    floor: int = 0

    def records(self):
        records, node = [], self
        while node is not None:
            records.append(
                {
                    "before_hash": node.before_hash,
                    "action": node.action,
                    "actor": node.actor,
                    "phase": node.phase,
                    "act": node.act,
                    "floor": node.floor,
                }
            )
            node = node.parent
        return records[::-1]


def from_records(records):
    node = None
    for record in records:
        node = Prefix(
            node,
            record["before_hash"],
            deepcopy(record["action"]),
            record["actor"],
            1 if node is None else node.length + 1,
            record.get("phase", ""),
            record.get("act", 0),
            record.get("floor", 0),
        )
    return node


def append(prefix, frame, candidate, actor):
    p = player(frame)
    return Prefix(
        prefix,
        state_key(frame),
        deepcopy(action_semantics(candidate)),
        actor,
        1 if prefix is None else prefix.length + 1,
        frame["public"]["phase"],
        p.get("act", 0),
        p.get("floor", 0),
    )


def player(frame):
    return next(e for e in frame["public"]["entities"] if e["entity_type"] == "player")


def card_value(card, deck):
    """Public-stat prior, not a claim to understand opaque card effects."""
    ident = card.get("content_id", "")
    kind = card.get("card_type", "").lower()
    if kind in ("curse", "status"):
        return -8.0
    if "STRIKE_" in ident:
        return -1.4 if card.get("upgraded") else -2.0
    if "DEFEND_" in ident:
        return -0.7 if card.get("upgraded") else -1.3
    stats = card.get("stats") or {}

    def number(key, default=0):
        value = stats.get(key, default)
        return float(value) if isinstance(value, (int, float)) else float(default)

    cost = max(0.5, float(card.get("cost", 1) or 0))
    value = 1.0 + {"rare": 1.0, "uncommon": 0.4}.get(card.get("rarity", "").lower(), 0)
    value += 0.5 * (kind == "power") + 0.4 * bool(card.get("upgraded"))
    value += min(
        4.0,
        max(0.0, number("StrengthPower")) * 0.9 + max(0.0, number("DexterityPower")),
    )
    hits = max(1.0, number("Repeat", 1))
    damage = number("Damage", number("CalculationBase"))
    strength = sum(float((c.get("stats") or {}).get("StrengthPower", 0)) for c in deck)
    if damage > 0:
        damage = (damage + 0.5 * strength) * hits
        if card.get("target_type") == "AllEnemies":
            damage *= 1.35
        value += max(-0.5, min(2.5, damage / cost / 5 - 1))
    if number("Block") > 0:
        value += max(-0.3, min(2.0, number("Block") / cost / 5 - 1))
    value += min(1.5, 0.35 * (number("WeakPower") + number("VulnerablePower")))
    value += min(1.5, 0.7 * max(0, number("Cards") - cost - 1))
    value -= max(0, len(deck) - 16) * 0.4
    value -= sum(c.get("content_id") == ident for c in deck) * 0.7
    return value


def map_route_cost(frame, start):
    """Minimum public room-risk estimate on the map DAG, including future rests."""
    nodes = {
        e["ref"]: e
        for e in frame["public"]["entities"]
        if e.get("entity_type") == "map_node" and e.get("ref")
    }
    if start not in nodes:
        return 0.0
    children = {ref: [] for ref in nodes}
    for edge in frame["public"].get("relations", []):
        if (
            edge.get("role") == "map_edge"
            and edge.get("source") in nodes
            and edge.get("target") in nodes
        ):
            children[edge["source"]].append(edge["target"])
    costs = {
        "MONSTER": 1.5,
        "ELITE": 3.0,
        "RESTSITE": -2.0,
        "REST": -2.0,
        "SHOP": 0.5,
        "MERCHANT": 0.5,
        "TREASURE": -0.1,
        "UNKNOWN": 1.0,
        "BOSS": 0.0,
        "ANCIENT": 0.0,
    }
    memo, visiting = {}, set()

    def visit(ref):
        if ref in memo:
            return memo[ref]
        if ref in visiting:
            raise ValueError("Public map contains a cycle")
        visiting.add(ref)
        kind = nodes[ref].get("content_id", "").upper()
        tail = min(
            (visit(child) for child in children[ref]),
            default=0 if kind == "BOSS" else 20.0,
        )
        memo[ref] = costs.get(kind, 1.5) + tail
        visiting.remove(ref)
        return memo[ref]

    return visit(start)


def preference(frame, candidate):
    entities = frame["public"]["entities"]
    refs = candidate.get("source_refs", [])
    source = next((e for e in entities if e.get("ref") in refs), {})
    deck = [
        e for e in entities if e["entity_type"] == "card" and e.get("zone") == "deck"
    ]
    p, verb = player(frame), candidate["verb"]
    hp = p["hp"] / max(1, p["max_hp"])
    content = source.get("content_id", "").upper()
    if verb == "TAKE_CARD_REWARD":
        return card_value(source, deck)
    if verb in ("TAKE_REWARD", "TAKE_RELIC", "OPEN_CHEST", "FINISH_SELECTION"):
        return 3.0
    if verb == "MOVE_TO_NODE":
        local = {
            "REST": 3 if hp < 0.65 else 1.5,
            "RESTSITE": 3 if hp < 0.65 else 1.5,
            "ELITE": -4 if hp < 0.6 else (1.5 if hp > 0.8 else 0),
            "TREASURE": 3,
            "SHOP": 1 if p.get("gold", 0) > 120 else -2,
            "MERCHANT": 1 if p.get("gold", 0) > 120 else -2,
            "UNKNOWN": 0.7,
        }.get(content, 0.0)
        return local - 0.75 * map_route_cost(frame, source.get("ref"))
    if verb == "CHOOSE_REST_OPTION":
        current = {
            e.get("ref")
            for e in entities
            if e.get("entity_type") == "map_node" and e.get("current")
        }
        bosses = {
            e.get("ref")
            for e in entities
            if e.get("entity_type") == "map_node"
            and e.get("content_id", "").upper() == "BOSS"
        }
        boss_next = any(
            r.get("role") == "map_edge"
            and r.get("source") in current
            and r.get("target") in bosses
            for r in frame["public"].get("relations", [])
        )
        if "HEAL" in content or "REST" in content:
            return 5 if hp < (0.9 if boss_next else 0.6) else -2
        if "SMITH" in content:
            return 3 if hp > 0.45 else 0
    if verb == "SELECT_ONE":
        operation = (
            (frame["public"].get("selection_context") or {})
            .get("operation", "")
            .lower()
        )
        if "remove" in operation or operation == "exhaust":
            return -card_value(source, deck)
        if "upgrade" in operation:
            return card_value(source, []) + (not source.get("upgraded"))
        return card_value(source, deck)
    if "BUY" in verb or "PURCHASE" in verb:
        if content.startswith("CARD."):
            offered = next(
                (
                    r["target"]
                    for r in frame["public"].get("relations", [])
                    if r.get("source") == source.get("ref")
                    and r.get("role") == "offers"
                ),
                None,
            )
            card = next((e for e in entities if e.get("ref") == offered), source)
            return 3 + card_value(card, deck)
        if content.startswith("RELIC."):
            return 4.5
        if content.startswith("POTION."):
            return 4.2 if hp < 0.7 else 3.0
        if "REMOVAL" in content:
            return 4.3
        return 1.5
    if "REMOVE" in verb:
        return 2
    if verb in ("CANCEL", "DISCARD_POTION", "ABANDON_RUN"):
        return -12
    if verb == "LEAVE_ROOM":
        return 4.0
    if verb in ("SKIP", "SKIP_REWARDS", "CHOOSE_REWARD_ALTERNATIVE"):
        return -0.5
    return 0.0


class ReplayWorker:
    def __init__(
        self,
        config,
        character,
        seed,
        budget_ms,
        deadline,
        max_steps,
        reuse_turn_plan=False,
        ascension=10,
        boss_budget_ms=5000,
    ):
        self.config, self.character, self.seed = config, character, seed
        self.budget_ms, self.deadline, self.max_steps = budget_ms, deadline, max_steps
        self.reuse_turn_plan = reuse_turn_plan
        self.ascension = ascension
        self.boss_budget_ms = boss_budget_ms
        self.engine = self.prefix = self.frame = None
        self.last_player = {}
        self.stop_path = None
        self.replayed = self.solver_steps = 0

    def close(self):
        if self.engine is not None:
            self.engine.close()
        self.engine = None

    def native_diagnostics(self):
        if self.engine is None or not hasattr(os, "pread"):
            return ""
        try:
            fd = self.engine.stderr.fileno()
            size = os.fstat(fd).st_size
            # Read without moving the file offset shared with the child process.
            return os.pread(fd, min(size, 65536), max(0, size - 65536)).decode(
                "utf-8", errors="replace"
            )
        except (OSError, ValueError):
            return ""

    def check(self):
        if self.stop_path and self.stop_path.exists():
            raise SearchLimit("STOP requested")
        if time.monotonic() >= self.deadline:
            raise SearchLimit("time budget exhausted")

    def settle(self):
        until = time.monotonic() + 30
        while True:
            self.check()
            if self.prefix and self.prefix.length >= self.max_steps:
                raise ReplayMismatch("maximum trajectory steps reached")
            validate_frame(self.frame)
            for entity in self.frame.get("public", {}).get("entities", []):
                if entity.get("entity_type") == "player":
                    self.last_player = entity
            if self.frame["boundary"] != "waiting":
                return self.frame
            if time.monotonic() > until:
                raise ReplayMismatch("native waiting boundary did not settle")
            self.frame = self.engine.send({"cmd": "advance_to_boundary"})

    def restore(self, prefix):
        if self.engine is not None and self.prefix is prefix:
            return self.settle()
        self.close()
        self.check()
        self.engine = SolverEngine(
            self.config,
            timeout=max(30, max(self.budget_ms, self.boss_budget_ms) / 1000 + 15),
        )
        self.frame = self.engine.reset(self.character, self.seed, self.ascension)
        self.prefix = None
        for record in prefix.records() if prefix else []:
            self.settle()
            if state_key(self.frame) != record["before_hash"]:
                raise ReplayMismatch("Native replay public state diverged")
            candidate = resolve(self.frame, record["action"])
            self.frame = self.engine.send(
                execution_command(self.frame, candidate["candidate_ref"])
            )
            self.replayed += 1
        self.prefix = prefix
        return self.settle()

    def execute(self, semantics, actor="mcts"):
        self.settle()
        candidate = resolve(self.frame, semantics)
        prefix = append(self.prefix, self.frame, candidate, actor)
        frame = self.engine.send(
            execution_command(self.frame, candidate["candidate_ref"])
        )
        self.prefix, self.frame = prefix, frame

    def combat_and_forced(self):
        in_combat = False
        while True:
            self.settle()
            if self.frame["boundary"] == "terminal":
                return self.frame
            phase = self.frame["public"]["phase"]
            if phase == "combat":
                in_combat = True
            elif phase not in ("card_select", "card_reward"):
                in_combat = False
            elif not in_combat:
                info = self.engine.send({"cmd": "solver_info"})
                if info.get("type") != "solver_info":
                    raise ReplayMismatch(str(info))
                in_combat = info["combat_in_progress"]
            if in_combat:
                result = self.engine.step(
                    self.frame,
                    budget_ms=self.budget_ms,
                    boss_budget_ms=self.boss_budget_ms,
                    potions=True,
                    reuse_turn_plan=self.reuse_turn_plan,
                )
                if result.get("type") == "solver_selection_required":
                    if phase not in ("card_select", "card_reward"):
                        raise ReplayMismatch(
                            "Solver requested selection outside a selection boundary"
                        )
                    if len(self.frame["legal"]["candidates"]) == 1:
                        self.execute(
                            action_semantics(self.frame["legal"]["candidates"][0]),
                            "forced",
                        )
                        continue
                    return self.frame
                if result.get("type") != "solver_step":
                    raise ReplayMismatch(str(result))
                candidate = next(
                    c
                    for c in self.frame["legal"]["candidates"]
                    if c["candidate_ref"] == result["candidate_ref"]
                )
                self.prefix = append(
                    self.prefix, self.frame, candidate, "combat_solver"
                )
                self.frame = result["frame"]
                self.solver_steps += 1
            elif len(self.frame["legal"]["candidates"]) == 1:
                self.execute(
                    action_semantics(self.frame["legal"]["candidates"][0]), "forced"
                )
            else:
                return self.frame


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temp.replace(path)
