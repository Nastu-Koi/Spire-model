"""Import Steam RunRecorder journals without replaying or inventing game actions.

Only pre-decision snapshots and recorder-enumerated candidates become inputs.
Raw next states, reflected UI objects, IDs, RNG and unrevealed rewards stay out.
"""
from collections import Counter
from copy import deepcopy
import glob
import hashlib
import json
from pathlib import Path

from .data import validate_run
from .protocol import CHARACTERS, SCHEMA, ProtocolError, clean_frame, fingerprint

RECORDER_SCHEMA = "sts2-run-recorder/v2"
IMPORT_VERSION = "run-recorder-import-v2"
PHASES = {"combat_play": "combat", "map_select": "map", "event_choice": "event",
          "rest_site": "rest_site", "shop": "shop", "treasure": "treasure",
          "reward_choice": "rewards", "card_select": "card_select",
          "card_reward": "card_reward", "bundle_select": "bundle_select",
          "crystal_sphere": "crystal_sphere"}
VERBS = {"play_card": "PLAY_CARD", "end_turn": "END_TURN", "use_potion": "USE_POTION",
         "discard_potion": "DISCARD_POTION", "select_map_node": "MOVE_TO_NODE",
         "choose_event_option": "CHOOSE_EVENT_OPTION", "choose_rest_option": "CHOOSE_REST_OPTION",
         "purchase": "BUY_ITEM", "leave_shop": "LEAVE_ROOM", "pick_relic": "TAKE_TREASURE_RELIC",
         "take_reward": "TAKE_REWARD", "skip_rewards": "LEAVE_REWARDS",
         "select_card": "SELECT_ONE", "select_bundle": "SELECT_BUNDLE",
         "select_reward_option": "TAKE_CARD_REWARD",
         "crystal_sphere_cell": "DIVINE_CELL",
         "skip": "SKIP", "cancel": "CANCEL", "finish_selection": "FINISH_SELECTION"}


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def opaque(content):
    return {"kind": "effect", "op": "OPAQUE_RULE", "content_id": content, "coverage": "opaque"}


def content_id(raw, category):
    value = raw.get("id")
    if value is None:
        return None
    return value if "." in value else category.upper() + "." + value


def card_fields(raw):
    result = {"entity_type": "card", "content_id": content_id(raw, "card"),
              "effect_coverage": "opaque"}
    for src, dst in {"current_cost": "cost", "cost": "base_cost", "costs_x": "x_cost",
                     "upgraded": "upgraded", "upgrade_level": "upgrade_level", "card_type": "card_type",
                     "rarity": "rarity", "star_cost": "star_cost", "target_type": "target_type",
                     "keywords": "keywords", "retain": "retain", "stars_x": "stars_x",
                     "exhaust_on_next_play": "exhaust_on_next_play"}.items():
        if src in raw:
            result[dst] = raw[src]
    result["stats"] = {k: v.get("base_value") for k, v in (raw.get("vars") or {}).items() if isinstance(v, dict)}
    if result["content_id"] == "CARD.MAD_SCIENCE":
        variant = raw.get("mad_science") or {}
        require(isinstance(variant.get("rider_effect"), str) and bool(variant["rider_effect"])
                and variant.get("card_type") == raw.get("card_type"),
                "Mad Science is missing its public type/rider variant")
        result["rider_effect"] = variant["rider_effect"]
    for key in ("enchantment", "affliction"):
        value = raw.get(key)
        if isinstance(value, dict):
            result[key] = value.get("id")
            if "amount" in value:
                result[key + "_amount"] = value["amount"]
    result["semantic_program"] = opaque(result["content_id"])
    return result


class PublicSnapshot:
    def __init__(self, state):
        players = state.get("players", [])
        require(len(players) == 1, "Only single-player recordings are supported")
        player = players[0]
        self.entities, self.relations, self.refs, self.keys = [], [], {}, {}
        run, combat = state["run"], state.get("combat") or {}
        pcs = player.get("combat") or {}
        self.add(player["creature"], "player", "player", {
            **{k: player["creature"].get(k) for k in ("hp", "max_hp", "block")},
            "character": player["character"], "act": run["act"], "floor": run["floor"],
            "gold": player["gold"], "energy": pcs.get("energy"), "max_energy": pcs.get("max_energy"),
            "stars": pcs.get("stars"), "round": combat.get("round"), "capacity": player.get("max_potions")})
        self.powers(player["creature"], "player")
        self.pile(player.get("deck", []), "deck")
        if combat.get("in_progress"):
            for source, zone in (("hand", "hand"), ("draw", "draw_pile"), ("discard", "discard_pile"),
                                 ("exhaust", "exhaust_pile"), ("play", "play")):
                self.pile(pcs.get(source, []), zone, ordered=source == "hand")
            for i, creature in enumerate(combat.get("enemies", [])):
                self.creature(creature, "enemy", f"creature:{i+1}")
            if player.get("osty"):
                self.creature(player["osty"], "summon", "osty")
            self.entities.append({"entity_type": "orb_slots", "capacity": pcs.get("orb_slots")})
            for i, orb in enumerate(pcs.get("orbs") or []):
                self.entities.append({"entity_type": "orb", "owner_ref": "player", "position": i,
                                      "content_id": content_id(orb, "orb"),
                                      "stats": {"passive": orb.get("passive_value"), "evoke": orb.get("evoke_value")}})
        for relic in player.get("relics", []):
            cid = content_id(relic, "relic")
            self.entities.append({"entity_type": "relic", "content_id": cid, "owner_ref": "player",
                                  "effect_coverage": "opaque", "semantic_program": opaque(cid)})
        for slot in player.get("potions", []):
            potion = slot.get("potion")
            if potion:
                cid = content_id(potion, "potion")
                self.add(potion, "potion", f"potion:{slot['slot']}", {
                    "content_id": cid, "slot": slot["slot"], "owner_ref": "player",
                    "effect_coverage": "opaque", "semantic_program": opaque(cid)})
        for node in state.get("map") or []:
            ref = self.map_ref(node)
            current = run.get("location") or {}
            self.entities.append({"entity_type": "map_node", "ref": ref, "content_id": node["type"],
                                  "floor": node["row"], "col": node["col"],
                                  "current": (node["row"], node["col"]) == (current.get("row"), current.get("col")),
                                  "visited": any(self.map_ref(x) == ref for x in run.get("visited", []))})
            self.relations.extend({"source": ref, "target": self.map_ref(child), "role": "map_edge"}
                                  for child in node.get("children", []))
        self.context(state["legal"])

    @staticmethod
    def map_ref(node):
        return f"map:{node['col']}:{node['row']}"

    def add(self, raw, kind, ref, fields):
        self.entities.append({"entity_type": kind, "ref": ref, **fields})
        if raw.get("instance_id") is not None:
            self.refs[raw["instance_id"]] = ref
        return ref

    def powers(self, creature, ref):
        for power in creature.get("powers") or []:
            self.entities.append({"entity_type": "power", "content_id": content_id(power, "power"),
                                  "owner_ref": ref, "stacks": power.get("amount")})

    def creature(self, raw, kind, ref):
        self.add(raw, kind, ref, {"content_id": content_id(raw, "monster"), "known": True,
                                 **{k: raw.get(k) for k in ("hp", "max_hp", "block")}})
        self.powers(raw, ref)
        for intent in raw.get("intents") or []:
            visible = {"entity_type": "intent", "owner_ref": ref, "intent": intent.get("type")}
            if intent.get("damage") is not None:
                visible.update(damage=intent["damage"], hits=intent.get("repeats", 1),
                               total_damage=intent["damage"] * intent.get("repeats", 1))
            self.entities.append(visible)

    def pile(self, cards, zone, ordered=False):
        # Canonicalize using public values only; instance IDs and hidden array order
        # never choose features, slots or positional encodings.
        cards = list(cards)
        if not ordered:
            cards.sort(key=lambda x: json.dumps(card_fields(x), sort_keys=True))
        for i, card in enumerate(cards):
            if card.get("instance_id") in self.refs:
                continue  # A deck card offered for removal is the same public object.
            ref = f"{zone}:{i}"
            self.add(card, "card", ref, {**card_fields(card), "zone": zone, "owner_ref": "player",
                                         **({"position": i} if ordered else {})})
        self.entities.append({"entity_type": "pile_summary", "zone": zone, "count": len(cards),
                              "complete": True, "order_known": ordered})

    def context(self, legal):
        scope, context = legal["scope"], legal.get("context")
        if scope == "event_choice":
            self.entities.append({"entity_type": "event", "content_id": "EVENT." + context["event_id"]})
            for i, option in enumerate(context["options"]):
                self.add(option, "event_option", f"option:{i}", {"content_id": option["text_key"],
                    "enabled": not option.get("locked", False), "effect_coverage": "opaque",
                    "semantic_program": opaque(option["text_key"])})
        elif scope in {"rest_site", "treasure"}:
            for item in context:
                i = item["index"]
                if scope == "rest_site":
                    kind, cid = "rest_option", item["option_id"]
                else:
                    kind, cid = "relic", content_id(item["relic"], "relic")
                ref = f"option:{i}"
                self.add(item, kind, ref, {"content_id": cid, "enabled": item.get("enabled", True),
                    "effect_coverage": "opaque", "semantic_program": opaque(cid)})
                self.keys[(scope, i)] = ref
        elif scope == "reward_choice":
            for i, item in enumerate(context["rewards"]):
                cid = item["reward_type"] + "Reward"
                value = item.get("value")
                if isinstance(value, dict) and item["reward_type"] in {"Potion", "Relic"}:
                    cid = content_id(value, item["reward_type"])
                fields = {"content_id": cid, "effect_coverage": "opaque", "semantic_program": opaque(cid)}
                if item["reward_type"] == "Gold":
                    fields["gold"] = value
                # CardReward.cards is unrevealed until its separate selection opens.
                self.add(item, "reward", f"reward:{i}", fields)
        elif scope == "shop":
            for i, item in enumerate(context):
                value = item["value"]
                card = (value.get("CreationResult") or {}).get("Card")
                obj = card or value.get("Model")
                typename = value.get("type", "").rsplit(".", 1)[-1]
                kind = "card" if card or typename == "MerchantCardEntry" else "relic" if typename == "MerchantRelicEntry" else "potion" if typename == "MerchantPotionEntry" else "service"
                sold_out = value.get("IsStocked") is False
                require(kind == "service" or obj is not None or sold_out, "Missing visible shop item")
                fields = card_fields(card) if card else {
                    "content_id": content_id(obj, kind) if obj else "card_removal" if kind == "service" else None}
                fields.update(price=value.get("Cost"), sold_out=not value.get("IsStocked", True),
                              zone="shop", effect_coverage="opaque", semantic_program=opaque(fields["content_id"]))
                self.add(item, "shop_item", f"shop:{i}", fields)
        elif scope == "card_select":
            require(isinstance(context, list), "Missing pre-selection card offer")
            # Rebind offered cards to the option zone, not to an internal/deck order.
            self.pile(context, "selection")
        elif scope == "card_reward":
            cards = context["cards"]
            self.pile(cards, "reward")
            for i, card in enumerate(cards):
                self.keys[(scope, i)] = self.refs[card["instance_id"]]
            for i, option in enumerate(context["alternatives"], len(cards)):
                cid = option.get("OptionId")
                require(isinstance(cid, str), "Unknown reward alternative")
                ref = f"alternative:{i}"
                self.entities.append({"entity_type": "reward_alternative", "ref": ref, "content_id": cid,
                                      "effect_coverage": "opaque", "semantic_program": opaque(cid)})
                self.keys[(scope, i)] = ref
        elif scope == "bundle_select":
            for bundle in context:
                ref = f"bundle:{bundle['index']}"
                self.entities.append({"entity_type": "bundle", "ref": ref,
                                      "cards": [card_fields(c) for c in bundle["cards"]]})
                self.keys[(scope, bundle["index"])] = ref
        elif scope == "crystal_sphere":
            self.entities.append({"entity_type": "event_state", "content_id": "CRYSTAL_SPHERE",
                                  "remaining_steps": context["remaining"]})
            for tool in ("Small", "Big"):
                self.entities.append({"entity_type": "tool", "ref": "tool:" + tool, "content_id": tool,
                    "cost": 1, "semantic_program": {"kind": "effect", "op": "REVEAL",
                    "shape": "cell" if tool == "Small" else "square", "radius": 0 if tool == "Small" else 1,
                    "clip_to_board": True, "count": 1}})
            for cell in context["cells"]:
                ref = f"cell:{cell['x']}:{cell['y']}"
                fields = {"entity_type": "board_cell", "ref": ref, "x": cell["x"], "y": cell["y"],
                          "revealed": not cell["hidden"], "known": not cell["hidden"]}
                if not cell["hidden"]:
                    item = cell.get("item")
                    fields["content_id"] = item.get("type", "unknown").rsplit(".", 1)[-1] if item else "empty"
                self.entities.append(fields)
                self.keys[(scope, (cell["x"], cell["y"]))] = ref
        elif scope not in {"combat_play", "map_select"}:
            raise ProtocolError("Unsupported recorded decision scope: " + scope)

    def action(self, raw, index, scope):
        command, args = raw["command"], raw.get("args", {})
        require(command in VERBS, "Unsupported recorded command: " + command)
        result = {"candidate_ref": f"c{index}", "decoder_slot_ref": f"a{index}", "verb": VERBS[command]}
        if command in {"pick_relic", "select_reward_option", "select_bundle"} and args.get("index") is None:
            result["verb"] = "LEAVE_ROOM" if command == "pick_relic" else "SKIP"
        else:
            source = next((args[k] for k in ("card_instance_id", "potion_instance_id", "option_instance_id",
                                            "reward_instance_id", "entry_instance_id") if k in args), None)
            if source is not None:
                require(source in self.refs, "Candidate refers to an unobserved source")
                result["source_ref"] = self.refs[source]
            elif command == "select_map_node":
                result["source_ref"] = self.map_ref(args)
                require(any(e.get("ref") == result["source_ref"] for e in self.entities), "Missing visible map node")
            elif command in {"choose_rest_option", "pick_relic", "select_reward_option", "select_bundle"}:
                result["source_ref"] = self.keys[(scope, args["index"])]
                if command == "select_reward_option" and result["source_ref"].startswith("alternative:"):
                    result["verb"] = "CHOOSE_REWARD_ALTERNATIVE"
            elif command == "crystal_sphere_cell":
                require(args["tool"] in {"Small", "Big"}, "Unknown crystal sphere tool")
                result["source_ref"] = "tool:" + args["tool"]
                result["target_refs"] = [self.keys[(scope, (args["x"], args["y"]))]]
        target = args.get("target_instance_id")
        if target is not None:
            require(target in self.refs, "Candidate refers to an unobserved target")
            result["target_refs"] = [self.refs[target]]
        return result


def decision_macro(decision, run_id, contract):
    require(decision.get("status") == "completed", "Decision was not completed")
    require(decision.get("state_boundary") in {"execution_or_semantic_entry", "input_submission", "selection_offer"},
            "Unknown pre-decision boundary")
    state = decision["state"]
    require(state["run"].get("ascension") == 10, "Recording is not A10")
    legal = state.get("legal") or {}
    require(legal.get("schema") == 1 and legal.get("status") == "complete", "Missing complete recorded legal set")
    actions = legal.get("actions", [])
    require(actions and len({fingerprint(x) for x in actions}) == len(actions), "Empty or duplicate recorded legal actions")
    scope = legal["scope"]
    require(scope in PHASES, "Unsupported recorded decision scope: " + scope)
    snapshot = PublicSnapshot(state)
    bank = [snapshot.action(x, i, scope) for i, x in enumerate(actions)]
    choice = decision.get("choice_key")
    selection = legal.get("selection")
    if decision.get("selection_path") is not None:
        return selection_macro(decision, run_id, contract, snapshot)
    if choice and choice["command"] == "select_cards":
        ids = choice["args"]["card_instance_ids"]
        # v0.3 only recorded the final collection. Multi-select cannot be treated
        # as independent one-card decisions or assigned an invented reveal order.
        require(selection and selection.get("min") == selection.get("max") == 1 and len(ids) == 1,
                "Selection requires a recorded prefix path; final set alone is insufficient")
        require(selection.get("cancelable") is False,
                "Legacy cancelable selection lacks explicit cancel candidates")
        choice = {"command": "select_card", "args": {"card_instance_id": ids[0]}}
    matches = [i for i, action in enumerate(actions) if action == choice]
    require(len(matches) == 1, "Recorded choice is not uniquely in the complete legal set")
    require(decision.get("legal_match", {}).get("status") == "matched", "Recorder did not verify choice legality")
    public = {"phase": PHASES[scope], "entities": snapshot.entities,
              "relations": snapshot.relations, "memory": [], "decoder_bank": bank}
    frame = {"type": "decision_frame", "contract": contract, "boundary": "decision",
             "routing": {"episode_id": run_id, "decision_id": decision["action_id"], "state_version": decision["order"]},
             "public": public, "legal": {"candidates": bank}, "events": []}
    return {"phase": PHASES[scope], "steps": [{"frame": clean_frame(frame),
            "candidate_ref": bank[matches[0]]["candidate_ref"], "forced": len(bank) == 1}]}


def selection_macro(decision, run_id, contract, snapshot):
    """Consume recorder-emitted prefix masks; verify them against the frozen offer."""
    legal = decision["state"]["legal"]
    path, metadata = decision["selection_path"], legal["selection"]
    require(path.get("mode") == "buffered" and path.get("reconstructed") is True
            and path.get("order_source") == "submitted_result", "Unknown selection path semantics")
    require(decision["choice_key"]["command"] == "select_cards", "Selection path label is not a card set")
    require(decision.get("legal_match", {}).get("status") == "matched", "Invalid recorded selection")
    offered = legal["actions"]
    require(all(a["command"] == "select_card" for a in offered), "Unexpected action in selection offer")
    ids = [a["args"]["card_instance_id"] for a in offered]
    result = decision["choice_key"]["args"]["card_instance_ids"]
    minimum, maximum, cancelable = metadata["min"], metadata["max"], metadata["cancelable"]
    require(type(minimum) is int and type(maximum) is int and 0 <= minimum <= maximum and type(cancelable) is bool,
            "Invalid selection bounds")
    require(len(result) == len(set(result)) and set(result) <= set(ids) and len(result) <= maximum
            and (len(result) >= minimum or (not result and cancelable)), "Invalid submitted selection")
    finish, cancel = {"command": "finish_selection", "args": {}}, {"command": "cancel", "args": {}}
    raw_bank = offered + [finish] + ([cancel] if cancelable else [])
    bank = [snapshot.action(a, i, "card_select") for i, a in enumerate(raw_bank)]
    require(len(path["steps"]) == len(result) + 1, "Incomplete selection prefix path")
    steps = []
    for revision, part in enumerate(path["steps"]):
        prefix = result[:revision]
        require(part["selected_ids"] == prefix, "Selection prefix disagrees with submitted order")
        expected = ([a for a in offered if a["args"]["card_instance_id"] not in prefix]
                    if len(prefix) < maximum else [])
        if len(prefix) >= minimum:
            expected += [finish]
        if cancelable:
            expected += [cancel]
        require(part["actions"] == expected, "Incomplete or invented selection prefix mask")
        label = ({"command": "select_card", "args": {"card_instance_id": result[revision]}}
                 if revision < len(result) else cancel if len(result) < minimum else finish)
        require(part["choice_key"] == label and label in expected, "Selection prefix label is illegal")
        candidates = [bank[raw_bank.index(a)] for a in expected]
        context = {"mode": "buffered", "min_total": minimum, "max_total": maximum,
                   "selected_refs": [snapshot.refs[x] for x in prefix], "selected_count": len(prefix),
                   "remaining_required": max(0, minimum - len(prefix)), "can_finish": finish in expected,
                   "can_cancel": cancel in expected, "can_skip": False, "repetition_allowed": False,
                   "order_matters": metadata.get("ordered", True)}
        frame = {"type": "decision_frame", "contract": contract, "boundary": "decision",
                 "routing": {"episode_id": run_id, "decision_id": f"{decision['action_id']}:{revision}",
                             "state_version": revision, "selection_id": decision["action_id"],
                             "selection_revision": revision, "base_public_version": decision["order"],
                             "action_bank_version": 0},
                 "public": {"phase": "card_select", "entities": snapshot.entities, "relations": snapshot.relations,
                            "memory": [], "decoder_bank": bank, "selection_context": context},
                 "legal": {"candidates": candidates}, "events": []}
        steps.append({"frame": clean_frame(frame), "candidate_ref": bank[raw_bank.index(label)]["candidate_ref"],
                      "forced": len(candidates) == 1})
    return {"phase": "card_select", "steps": steps, "label_path": "reconstructed_from_submitted_result"}


def flatten(decision):
    yield decision
    for child in decision.get("choices", []):
        yield from flatten(child)


def verify_teachers(decisions):
    by_id = {d["action_id"]: d for d in decisions}
    def public_teacher(decision, seen):
        if decision.get("actor") != "human" or decision["action_id"] in seen:
            return False
        evidence = decision.get("actor_evidence", "")
        if evidence == "game_ui_call_stack":
            return True
        if evidence.startswith("parent_action:"):
            parent = by_id.get(evidence.removeprefix("parent_action:"))
            return parent is not None and public_teacher(parent, seen | {decision["action_id"]})
        return False
    require(all(public_teacher(d, set()) for d in decisions),
            "Teacher visibility is unverified (solver/unknown actor); public-input cleaning cannot verify the teacher")


def read_journal(path):
    """Validate the append-only envelope before considering any sample."""
    rows, digest = [], hashlib.sha256()
    with Path(path).open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            require(raw.endswith(b"\n"), f"Truncated journal line {number}")
            row = json.loads(raw)
            require(isinstance(row, dict) and row.get("schema") == RECORDER_SCHEMA,
                    f"Unsupported recorder schema on line {number}")
            require(row.get("seq") == number, f"Journal sequence gap on line {number}")
            require(row.get("run_id") and row.get("segment_id"), "Missing journal identity")
            if rows:
                require((row["run_id"], row["segment_id"]) == (rows[0]["run_id"], rows[0]["segment_id"]),
                        "Mixed journal identities")
            rows.append(row)
    require(rows, "Empty recording")
    return rows, digest.hexdigest()


def convert_journal(path, *, bc_only=False, recorder_version=None):
    rows, digest = read_journal(path)
    require(rows[0]["kind"] == "segment_start", "Missing segment start")
    start = rows[0]["data"]
    if recorder_version is not None:
        require(start.get("recorder_version") == recorder_version, "Recorder version does not match filter")
    endings = [row["data"] for row in rows if row["kind"] == "run_ended"]
    if not bc_only:
        require(start.get("begins_at_run_start") is True, "Resumed fragment; complete-run continuity is unverified")
        require(rows[-1]["kind"] == "segment_end", "Recording is active or truncated; no segment end")
        require(rows[-1]["data"].get("uncommitted_operations") == 0, "Recording has uncommitted decisions")
        require(rows[-1]["data"].get("state_capture_errors", 0) == 0, "Recording contains failed state captures")
        require(len(endings) == 1 and type(endings[0].get("victory")) is bool, "Run has no unique terminal outcome")
        terminal = endings[0].get("state", {}).get("run", {})
        require(terminal.get("ended") is True and terminal.get("victory") == endings[0]["victory"]
                and terminal.get("ascension") == 10, "Terminal snapshot does not confirm outcome")
        require(rows[-1]["data"].get("run_ended") is True, "Segment end does not confirm run completion")
        allowed = {"segment_start", "segment_end", "run_ended", "decision_committed"}
        require(all(row["kind"] in allowed for row in rows), "Recording contains errors or an unsupported event format")
        require(sum(row["kind"] == "segment_start" for row in rows) == 1, "Repeated segment start")
    state = start["state"]
    require(state["run"].get("ascension") == 10, "Recording is not A10")
    require(len(state["players"]) == 1, "Only single-player recordings are supported")
    character = next((x for x in CHARACTERS if x.upper() == state["players"][0]["character"]), None)
    require(character is not None, "Unsupported character")
    decisions = [item for row in rows if row["kind"] == "decision_committed" for item in flatten(row["data"])]
    require(decisions, "Recording has no decisions")
    decisions.sort(key=lambda d: d["order"])
    require(len({d["action_id"] for d in decisions}) == len(decisions), "Repeated decision identity")
    require(len({d["order"] for d in decisions}) == len(decisions), "Ambiguous decision order")
    actors = Counter(d.get("actor", "unknown") for d in decisions)
    # A solver actor label does not establish what information its search used.
    if not bc_only:
        verify_teachers(decisions)
    contract = {"adapter_version": IMPORT_VERSION, "observation_schema": "public-state-v1",
                "action_schema": "candidate-v0", "fixed_ascension": 10, "training_ready": True,
                "game_version": start["game_version"], "recorder_version": start["recorder_version"],
                "game_assembly_mvid": next((a["mvid"] for a in start.get("loaded_assemblies", []) if a["name"] == "sts2"), None),
                "effect_coverage": "typed-public-fields-with-explicit-opaque-rules",
                "action_boundary": "recorded-native-ui", "purpose": "offline_behavior_cloning"}
    require(contract["game_assembly_mvid"], "Missing recorded game assembly identity")
    macros, automatic, rejected = [], 0, []
    for decision in decisions:
        try:
            require(decision["state"]["players"][0]["character"] == character.upper(), "Character changed within recording")
            require(str(decision["state"]["run"].get("seed")) == str(start["seed"]), "Seed changed within recording")
            macro = decision_macro(decision, rows[0]["run_id"], contract)
        except (ProtocolError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            if not bc_only:
                raise
            rejected.append({"action_id": decision.get("action_id"), "reason": str(exc)})
            continue
        if all(s["forced"] for s in macro["steps"]):
            automatic += len(macro["steps"])
        else:
            macros.append(macro)
    require(macros, "Recording contains no branching decisions")
    run = {"schema": SCHEMA, "source": "demonstration", "teacher_visibility": "public",
           "status": "complete", "victory": endings[0]["victory"] if not bc_only else None, "ascension": 10,
           "character": character, "run_id": rows[0]["run_id"], "seed": str(start["seed"]),
           "contract": contract, "macros": macros, "automatic_steps": automatic,
           "provenance": {"importer": IMPORT_VERSION, "raw_sha256": digest,
                          "segment_id": rows[0]["segment_id"], "actors": dict(actors)}}
    if bc_only:
        # These are supervised decisions, never a claim of complete on-policy play.
        run.update(source="recorder_bc", teacher_visibility="unverified",
                   status="partial", victory=None,
                   run_id=rows[0]["run_id"] + ":" + rows[0]["segment_id"])
        run["provenance"].update(bc_only=True, original_run_id=rows[0]["run_id"],
                                 rejected_decisions=rejected, decisions_seen=len(decisions))
    return validate_run(run)


def import_recordings(paths, output, *, bc_only=False, recorder_version=None):
    """Create an immutable import batch. Raw files remain untouched."""
    files = set()
    for pattern in paths:
        matches = glob.glob(str(pattern))
        require(matches, "No recording matches: " + str(pattern))
        for match in matches:
            path = Path(match)
            files.update(path.glob("*.jsonl") if path.is_dir() else [path])
    require(files, "No JSONL recordings found")
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    counts, reasons, identities = Counter(), Counter(), set()
    with (directory / "accepted.jsonl").open("x", encoding="utf-8") as good, \
            (directory / "quarantine.jsonl").open("x", encoding="utf-8") as bad:
        for path in sorted(files):
            try:
                if recorder_version is not None:
                    with path.open() as stream:
                        header = json.loads(next(stream, ""))
                    if header.get("data", {}).get("recorder_version") != recorder_version:
                        counts["version_filtered_files"] += 1
                        continue
                run = convert_journal(path, bc_only=bc_only, recorder_version=recorder_version)
                require(run["run_id"] not in identities, "Duplicate complete-run identity")
                identities.add(run["run_id"])
                good.write(json.dumps(run, ensure_ascii=False, allow_nan=False) + "\n")
                counts["accepted_runs"] += 1
                counts["macros"] += len(run["macros"])
                counts["rejected_decisions"] += len(run["provenance"].get("rejected_decisions", []))
                for macro in run["macros"]:
                    counts[run["character"] + "/" + macro["phase"]] += 1
            except (ProtocolError, ValueError, KeyError, TypeError, OSError, IndexError, AttributeError) as exc:
                reason = str(exc)
                reasons[reason] += 1
                counts["quarantined_files"] += 1
                bad.write(json.dumps({"path": str(path.resolve()), "reason": reason}, ensure_ascii=False) + "\n")
    summary = {"importer": IMPORT_VERSION, "bc_only": bc_only, "recorder_version_filter": recorder_version, "files": len(files), "accepted_runs": 0, **dict(counts),
               "quarantine_reasons": dict(reasons), "accepted": str(directory / "accepted.jsonl"),
               "quarantine": str(directory / "quarantine.jsonl")}
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
