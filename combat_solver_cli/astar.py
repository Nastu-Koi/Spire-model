"""Budgeted weighted A* over native run histories; combat is a solver macro."""

import heapq
import itertools
import json
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from model.protocol import (
    CHARACTERS,
    action_semantics,
    execution_command,
    fingerprint,
    validate_frame,
)

from .client import InfrastructureError, SolverEngine
from .diversity import RouteDiversity


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


def prefix_progress(prefix):
    return -1 if prefix is None else max(0, prefix.act - 1) * 20 + prefix.floor


def pop_frontier(frontier, backtrack_before=None, diversity=None):
    """Pop by A* score, optionally forcing an earlier unexplored decision."""
    if backtrack_before is None and diversity is None:
        return heapq.heappop(frontier)
    eligible = [
        i
        for i, node in enumerate(frontier)
        if backtrack_before is None or prefix_progress(node[3]) <= backtrack_before
    ]
    if not eligible:
        eligible = range(len(frontier))
    index = min(
        eligible,
        key=lambda i: diversity.rank(frontier[i]) if diversity else frontier[i],
    )
    item = frontier.pop(index)
    heapq.heapify(frontier)
    return item


def failure_backtrack(act, floor, failures):
    """Diversify after every native defeat, with independent depth per location."""
    if type(act) is not int or type(floor) is not int:
        return None
    key = f"{act}:{floor}"
    failures[key] = failures.get(key, 0) + 1
    progress = max(0, act - 1) * 20 + floor
    return max(0, progress - (3 + 2 * (failures[key] - 1)))


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


def heuristic(frame):
    p = player(frame)
    entities = frame["public"]["entities"]
    nodes = [e for e in entities if e["entity_type"] == "map_node"]
    act, floor = p.get("act", 1), p.get("floor", 0)
    remaining = (
        max(0, max((n.get("floor", 0) for n in nodes), default=16) + 1 - floor)
        + max(0, 3 - act) * 20
    )
    hp = p["hp"] / max(1, p["max_hp"])
    # Native next-act entry heals at the ancient. Keep its expected value continuous
    # across the preceding boss reward and the next act's floor-zero map.
    boss_rewards = (
        act < 3
        and frame["public"]["phase"] != "combat"
        and any(
            n.get("current") and n.get("content_id", "").lower() == "boss"
            for n in nodes
        )
    )
    if boss_rewards or (act > 1 and floor == 0):
        hp = max(hp, 0.75)
    deck = [
        e for e in entities if e["entity_type"] == "card" and e.get("zone") == "deck"
    ]
    card_scores = sorted((max(0, card_value(c, [])) for c in deck), reverse=True)
    strength = sum(card_scores[:20]) / 4 + sum(
        e["entity_type"] == "relic" for e in entities
    )
    deck_bloat = max(0, len(deck) - 24) * 0.75
    phase_cost = {
        "shop": 4.0,
        "card_select": 3.0,
        "card_reward": 2.0,
        "rewards": 2.0,
        "rest_site": 2.0,
    }.get(frame["public"]["phase"], 0.0)
    return (
        max(
            0,
            3 * remaining + 6 * (1 - hp) + deck_bloat - min(strength, remaining * 0.25),
        )
        + phase_cost
    )


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

    def execute(self, semantics, actor="astar"):
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


def save_frontier(
    path, frontier, character, seed, weight, ascension=10, search_state=None
):
    indices, prefixes = {}, []

    def index(node):
        chain = []
        while node is not None and id(node) not in indices:
            chain.append(node)
            node = node.parent
        parent = -1 if node is None else indices[id(node)]
        for node in reversed(chain):
            indices[id(node)] = len(prefixes)
            prefixes.append(
                {
                    "parent": parent,
                    "before_hash": node.before_hash,
                    "action": node.action,
                    "actor": node.actor,
                    "length": node.length,
                    "phase": node.phase,
                    "act": node.act,
                    "floor": node.floor,
                }
            )
            parent = len(prefixes) - 1
        return parent

    queue = [
        [f, tie, g, index(prefix), action] for f, tie, g, prefix, action in frontier
    ]
    write_json(
        path,
        {
            "schema": "astar-frontier-v2",
            "heuristic_version": 3,
            "character": character,
            "seed": seed,
            "ascension": ascension,
            "weight": weight,
            "prefixes": prefixes,
            "frontier": queue,
            "search_state": search_state or {},
        },
    )


def load_frontier(path, character, seed, weight, ascension=10, with_state=False):
    data = json.loads(Path(path).read_text())
    if (
        data["schema"],
        data["character"],
        data["seed"],
        data.get("ascension", 10),
        data["weight"],
    ) != ("astar-frontier-v2", character, seed, ascension, weight):
        raise ValueError("Checkpoint identity or search weight differs")
    prefixes = []
    for p in data["prefixes"]:
        parent = None if p["parent"] == -1 else prefixes[p["parent"]]
        prefixes.append(
            Prefix(
                parent,
                p["before_hash"],
                p["action"],
                p["actor"],
                p["length"],
                p.get("phase", ""),
                p.get("act", 0),
                p.get("floor", 0),
            )
        )
    queue = [
        (f, tie, g, None if i == -1 else prefixes[i], action)
        for f, tie, g, i, action in data["frontier"]
    ]
    heapq.heapify(queue)
    return (queue, data.get("search_state", {})) if with_state else queue


def search(
    config,
    character,
    seed,
    output,
    *,
    budget_ms=1000,
    weight=5.0,
    max_expansions=2000,
    max_seconds=3600,
    max_steps=10000,
    resume=None,
    prefix_path=None,
    reuse_turn_plan=False,
    rollout_decisions=0,
    ascension=10,
    search_lanes=1,
    boss_budget_ms=5000,
    early_route_diversity=True,
):
    if search_lanes not in (1, 2, 4):
        raise ValueError("search_lanes must be 1, 2 or 4")
    if character not in CHARACTERS:
        raise ValueError("Unsupported character")
    if type(ascension) is not int or not 0 <= ascension <= 10:
        raise ValueError("ascension must be an integer from 0 to 10")
    if any(
        not math.isfinite(x) or x <= 0
        for x in (budget_ms, weight, max_expansions, max_seconds, max_steps)
    ):
        raise ValueError("Search budgets must be finite and positive")
    if type(rollout_decisions) is not int or rollout_decisions < 0:
        raise ValueError("rollout_decisions must be a nonnegative integer")
    if type(boss_budget_ms) is not int or not 1 <= boss_budget_ms <= 120000:
        raise ValueError("boss_budget_ms must be an integer from 1 to 120000")
    if budget_ms > 120000:
        raise ValueError("CombatSolver budget_ms cannot exceed 120000")
    if resume and prefix_path:
        raise ValueError("Choose either a frontier or a prefix")
    if search_lanes > 1:
        from .lanes import search_parallel

        return search_parallel(
            config,
            character,
            seed,
            output,
            budget_ms=budget_ms,
            weight=weight,
            max_expansions=max_expansions,
            max_seconds=max_seconds,
            max_steps=max_steps,
            resume=resume,
            prefix_path=prefix_path,
            reuse_turn_plan=reuse_turn_plan,
            rollout_decisions=rollout_decisions,
            ascension=ascension,
            search_lanes=search_lanes,
            boss_budget_ms=boss_budget_ms,
            early_route_diversity=early_route_diversity,
        )
    resume_state = {}
    if resume:
        frontier, resume_state = load_frontier(
            resume, character, seed, weight, ascension, with_state=True
        )
    else:
        frontier = [(0.0, 0, 0.0, None, None)]
    if prefix_path:
        data = json.loads(Path(prefix_path).read_text())
        if (data["character"], data["seed"], data.get("ascension", 10)) != (
            character,
            seed,
            ascension,
        ):
            raise ValueError("Prefix identity differs")
        frontier = [(0.0, 0, 0.0, from_records(data["records"]), None)]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    serial = itertools.count(max((x[1] for x in frontier), default=-1) + 1)
    started = time.monotonic()
    worker = ReplayWorker(
        config,
        character,
        seed,
        budget_ms,
        started + max_seconds,
        max_steps,
        reuse_turn_plan,
        ascension,
        boss_budget_ms,
    )
    worker.stop_path = output / "STOP"
    expanded = deaths = errors = 0
    best, winner, active = (0, 0), None, None
    status, reason = "budget_exhausted", None
    probe_next, probe_left = None, 0
    backtrack_before = resume_state.get("backtrack_before")
    boss_failures = int(resume_state.get("boss_failures", 0))
    failure_counts = dict(resume_state.get("failure_counts", {}))
    diversity = (
        RouteDiversity(resume_state.get("route_visits"))
        if early_route_diversity
        else None
    )

    def checkpoint():
        save_frontier(
            output / "frontier.json",
            frontier,
            character,
            seed,
            weight,
            ascension,
            {
                "backtrack_before": backtrack_before,
                "boss_failures": boss_failures,
                "failure_counts": failure_counts,
                "route_visits": diversity.visits
                if diversity
                else resume_state.get("route_visits", {}),
            },
        )

    with (output / "search.jsonl").open("w") as journal:

        def log(event):
            event.update(character=character, seed=seed, ascension=ascension)
            journal.write(json.dumps(event, ensure_ascii=False) + "\n")
            journal.flush()

        try:
            while frontier and expanded < max_expansions:
                worker.check()
                using_probe = probe_next is not None and probe_left > 0
                if using_probe:
                    index = next(
                        i for i, node in enumerate(frontier) if node[1] == probe_next
                    )
                    active = frontier.pop(index)
                    heapq.heapify(frontier)
                    probe_left -= 1
                else:
                    active = pop_frontier(frontier, backtrack_before, diversity)
                    if diversity:
                        diversity.begin()
                    backtrack_before = None
                    probe_left = rollout_decisions
                if diversity:
                    diversity.observe(active)
                probe_next = None
                expanded += 1
                f, _, g, parent, pending = active
                try:
                    worker.restore(parent)
                    if pending is not None:
                        worker.execute(
                            pending, "astar_rollout" if using_probe else "astar"
                        )
                    frame = worker.combat_and_forced()
                    if frame["boundary"] == "terminal":
                        if frame["public"]["outcome"]["victory"]:
                            winner = worker.prefix
                            status = "candidate_victory"
                            break
                        deaths += 1
                        dead = next(
                            (
                                e
                                for e in frame["public"].get("entities", [])
                                if e.get("entity_type") == "player"
                            ),
                            getattr(worker, "last_player", {}),
                        )
                        dead_act, dead_floor = dead.get("act"), dead.get("floor")
                        if (
                            isinstance(dead_act, int)
                            and isinstance(dead_floor, int)
                            and dead_floor >= 15
                        ):
                            boss_failures += 1
                        backtrack_before = failure_backtrack(
                            dead_act, dead_floor, failure_counts
                        )
                        log(
                            {
                                "expanded": expanded,
                                "result": "native_defeat",
                                "act": dead_act,
                                "floor": dead_floor,
                                "prefix_steps": worker.prefix.length
                                if worker.prefix
                                else 0,
                                "boss_failures": boss_failures,
                                "backtrack_before": backtrack_before,
                            }
                        )
                        path = (
                            output
                            / f"death-act-{dead.get('act')}-floor-{dead.get('floor')}.json"
                        )
                        if not path.exists():
                            write_json(
                                path,
                                {
                                    "character": character,
                                    "seed": seed,
                                    "ascension": ascension,
                                    "records": worker.prefix.records()
                                    if worker.prefix
                                    else [],
                                    "outcome": frame["public"]["outcome"],
                                },
                            )
                        active = None
                        continue
                    p = player(frame)
                    progress = (p.get("act", 1), p.get("floor", 0))
                    if progress > best:
                        best = progress
                        write_json(
                            output / "best_prefix.json",
                            {
                                "character": character,
                                "seed": seed,
                                "ascension": ascension,
                                "records": worker.prefix.records()
                                if worker.prefix
                                else [],
                                "progress": best,
                            },
                        )
                    children = []
                    observation_key = state_key(frame)
                    # Stable seed-dependent tie order prevents every seed choosing option zero
                    # when opaque event/relic options have identical heuristic scores.
                    candidates = sorted(
                        frame["legal"]["candidates"],
                        key=lambda c: fingerprint(
                            [str(seed), observation_key, action_semantics(c)]
                        ),
                    )
                    for candidate in candidates:
                        if candidate["verb"] == "ABANDON_RUN":
                            continue
                        estimate = max(
                            0, heuristic(frame) - 0.25 * preference(frame, candidate)
                        )
                        child = (
                            g + 0.1 + weight * estimate,
                            next(serial),
                            g + 0.1,
                            worker.prefix,
                            action_semantics(candidate),
                        )
                        heapq.heappush(frontier, child)
                        children.append(child)
                    if probe_left > 0 and children:
                        probe_next = min(children)[1]
                    event = {
                        "expanded": expanded,
                        "frontier": len(frontier),
                        "act": progress[0],
                        "floor": progress[1],
                        "hp": p["hp"],
                        "max_hp": p["max_hp"],
                        "deck_size": sum(
                            e.get("entity_type") == "card" and e.get("zone") == "deck"
                            for e in frame["public"]["entities"]
                        ),
                        "phase": frame["public"]["phase"],
                        "f": f,
                        "prefix_steps": worker.prefix.length if worker.prefix else 0,
                    }
                    log(event)
                    if expanded % 10 == 0:
                        print(json.dumps(event), flush=True)
                except (InfrastructureError, SearchLimit):
                    raise
                except (RuntimeError, ValueError, KeyError) as exc:
                    errors += 1
                    log({"expanded": expanded, "error": str(exc)})
                    directory = output / "unresolved"
                    directory.mkdir(exist_ok=True)
                    path = directory / (fingerprint(str(exc))[:16] + ".json")
                    if not path.exists():
                        write_json(
                            path,
                            {
                                "character": character,
                                "seed": seed,
                                "ascension": ascension,
                                "error": str(exc),
                                "records": parent.records() if parent else [],
                                "pending_action": pending,
                            },
                        )
                        diagnostics = getattr(
                            worker, "native_diagnostics", lambda: ""
                        )()
                        if diagnostics:
                            path.with_suffix(".stderr.log").write_text(diagnostics)
                    worker.close()
                active = None
                if expanded % 25 == 0:
                    checkpoint()
            if not frontier and winner is None:
                status = (
                    "frontier_exhausted_with_unresolved"
                    if errors
                    else "frontier_exhausted"
                )
        except InfrastructureError as exc:
            status, reason = "infrastructure_error", str(exc)
        except (SearchLimit, KeyboardInterrupt) as exc:
            reason = str(exc) or "interrupted"
            if isinstance(exc, KeyboardInterrupt) or "STOP" in reason:
                status = "stopped"
        finally:
            worker.close()
            if active is not None and winner is None:
                heapq.heappush(frontier, active)
            if winner is None:
                checkpoint()
    summary = {
        "status": status,
        "character": character,
        "seed": seed,
        "ascension": ascension,
        "expanded": expanded,
        "deaths": deaths,
        "unresolved_branches": errors,
        "frontier": len(frontier),
        "best_progress": best,
        "solver_steps": worker.solver_steps,
        "replayed_steps": worker.replayed,
        "seconds": time.monotonic() - started,
        "algorithm": "weighted_astar_early_routes_v3"
        if diversity
        else "weighted_astar_location_backjump_v2",
        "heuristic_weight": weight,
        "optimality_proven": False,
        "budget_ms": budget_ms,
        "reuse_turn_plan": reuse_turn_plan,
        "rollout_decisions": rollout_decisions,
        "boss_failures": boss_failures,
        "failure_counts": failure_counts,
        "stop_reason": reason,
        "boss_budget_ms": boss_budget_ms,
        "early_route_diversity": early_route_diversity,
        "route_visits": diversity.visits if diversity else {},
    }
    if winner is not None:
        write_json(
            output / "winning_prefix.json",
            {
                "character": character,
                "seed": seed,
                "ascension": ascension,
                "records": winner.records(),
                "search": {
                    "algorithm": summary["algorithm"],
                    "weight": weight,
                    "budget_ms": budget_ms,
                    "boss_budget_ms": boss_budget_ms,
                    "early_route_diversity": early_route_diversity,
                    "reuse_turn_plan": reuse_turn_plan,
                    "rollout_decisions": rollout_decisions,
                },
            },
        )
        try:
            from .trajectory import verify_and_export

            verify_and_export(config, output / "winning_prefix.json", output)
            summary["status"] = "verified_victory"
            (output / "frontier.json").unlink(missing_ok=True)
        except Exception as exc:
            summary.update(status="replay_failed", verification_error=str(exc))
            # Verification may fail transiently; retain the winning node and alternatives.
            if active is not None:
                heapq.heappush(frontier, active)
            checkpoint()
    write_json(output / "summary.json", summary)
    return summary
