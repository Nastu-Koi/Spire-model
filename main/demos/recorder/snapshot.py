"""The public state of a RunRecorder journal entry as the model sees it (`PublicSnapshot`):
entities, fields and candidates built from the recorder's pre-decision snapshot and its
enumerated candidates. Raw next states, reflected UI objects, IDs, RNG and unrevealed
rewards stay out."""

import json

from main.env.public_history import draw_relations, recorded_memory
from main.env.protocol import ProtocolError


VERBS = {
    "play_card": "PLAY_CARD",
    "end_turn": "END_TURN",
    "use_potion": "USE_POTION",
    "discard_potion": "DISCARD_POTION",
    "select_map_node": "MOVE_TO_NODE",
    "choose_event_option": "CHOOSE_EVENT_OPTION",
    "choose_rest_option": "CHOOSE_REST_OPTION",
    "purchase": "BUY_ITEM",
    "leave_shop": "LEAVE_ROOM",
    "open_chest": "OPEN_CHEST",
    "pick_relic": "TAKE_TREASURE_RELIC",
    "take_reward": "TAKE_REWARD",
    "take_card_reward": "TAKE_CARD_REWARD",
    "take_reward_alternative": "CHOOSE_REWARD_ALTERNATIVE",
    "skip_rewards": "LEAVE_REWARDS",
    "select_card": "SELECT_ONE",
    "select_bundle": "SELECT_BUNDLE",
    "select_reward_option": "TAKE_CARD_REWARD",
    "crystal_sphere_cell": "DIVINE_CELL",
    "skip": "SKIP",
    "cancel": "CANCEL",
    "finish_selection": "FINISH_SELECTION",
}


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def opaque(content):
    return {
        "kind": "effect",
        "op": "OPAQUE_RULE",
        "content_id": content,
        "coverage": "opaque",
    }


def content_id(raw, category):
    value = raw.get("id")
    if value is None:
        return None
    return value if "." in value else category.upper() + "." + value


def card_fields(raw):
    """A recorded card with the fields of the engine's card export: the same names,
    and null where the engine publishes null."""
    result = {
        "entity_type": "card",
        "content_id": content_id(raw, "card"),
        "effect_coverage": "opaque",
        "rider_effect": None,
    }
    for src, dst in {
        "current_cost": "cost",
        "costs_x": "x_cost",
        "upgraded": "upgraded",
        "upgrade_level": "upgrade_level",
        "card_type": "card_type",
        "rarity": "rarity",
        "base_star_cost": "base_star_cost",
        "target_type": "target_type",
        "retain": "retain",
        "stars_x": "stars_x",
        "exhaust_on_next_play": "exhaust_on_next_play",
    }.items():
        if src in raw:
            result[dst] = raw[src]
    # The engine's star cost is the current one; recordings also carry the printed one.
    for src in ("current_star_cost", "star_cost"):
        if src in raw:
            result["star_cost"] = raw[src]
            break
    if "keywords" in raw:
        result["keywords"] = sorted(raw["keywords"] or [])
    result["stats"] = base_values(raw)
    if result["content_id"] == "CARD.MAD_SCIENCE":
        variant = raw.get("mad_science") or {}
        require(
            isinstance(variant.get("rider_effect"), str)
            and bool(variant["rider_effect"])
            and variant.get("card_type") == raw.get("card_type"),
            "Mad Science is missing its public type/rider variant",
        )
        result["rider_effect"] = variant["rider_effect"]
    for key in ("enchantment", "affliction"):
        value = raw.get(key)
        result[key] = content_id(value, key) if isinstance(value, dict) else None
        result[key + "_amount"] = value.get("amount") if isinstance(value, dict) else None
    result["semantic_program"] = opaque(result["content_id"])
    return result


def base_values(raw):
    """Displayed variables of a recorded model, as the engine exports them."""
    return {
        k: v.get("base_value")
        for k, v in (raw.get("vars") or {}).items()
        if isinstance(v, dict)
    }


class PublicSnapshot:
    def __init__(self, state):
        players = state.get("players", [])
        require(len(players) == 1, "Only single-player recordings are supported")
        player = players[0]
        self.entities, self.relations, self.refs, self.keys = [], [], {}, {}
        run, combat = state["run"], state.get("combat") or {}
        pcs = player.get("combat") or {}
        # The game keeps the last fight's energy and stars after it ends; they are
        # shown, and published by the engine's anchors, only during a fight.
        fighting = pcs if combat.get("in_progress") else {}
        self.add(
            player["creature"],
            "player",
            "player",
            {
                **{k: player["creature"].get(k) for k in ("hp", "max_hp", "block")},
                "character": player["character"],
                "ascension": run.get("ascension"),
                "act": run["act"],
                "floor": run["floor"],
                "gold": player["gold"],
                "energy": fighting.get("energy"),
                "max_energy": fighting.get("max_energy"),
                "stars": fighting.get("stars"),
                "round": combat.get("round") if fighting else None,
                "capacity": player.get("max_potions"),
                # Absent from recordings made before the mod exported it: unknown.
                "free_travel": run.get("free_travel"),
                "lost_hp_this_turn": fighting.get("lost_hp_this_turn"),
            },
        )
        self.powers(player["creature"], "player")
        self.pile(player.get("deck", []), "deck")
        if combat.get("in_progress"):
            for source, zone in (
                ("hand", "hand"),
                ("draw", "draw_pile"),
                ("discard", "discard_pile"),
                ("exhaust", "exhaust_pile"),
            ):
                self.pile(pcs.get(source, []), zone, ordered=source == "hand")
            for i, creature in enumerate(combat.get("enemies", [])):
                self.creature(creature, "enemy", f"creature:{i + 1}")
            if player.get("osty"):
                self.creature(player["osty"], "summon", "osty")
            self.entities.append(
                {"entity_type": "orb_slots", "capacity": pcs.get("orb_slots")}
            )
            for i, orb in enumerate(pcs.get("orbs") or []):
                self.entities.append(
                    {
                        "entity_type": "orb",
                        "owner_ref": "player",
                        "position": i,
                        "content_id": content_id(orb, "orb"),
                        "stats": {
                            "passive": orb.get("passive_value"),
                            "evoke": orb.get("evoke_value"),
                        },
                    }
                )
        for relic in player.get("relics", []):
            cid = content_id(relic, "relic")
            entity = {
                "entity_type": "relic",
                "content_id": cid,
                "owner_ref": "player",
                "stats": base_values(relic),
                "effect_coverage": "opaque",
                "semantic_program": opaque(cid),
            }
            if relic.get("show_counter"):
                entity["counter"] = relic.get("counter")
            self.entities.append(entity)
        for slot in player.get("potions", []):
            potion = slot.get("potion")
            if potion:
                cid = content_id(potion, "potion")
                self.add(
                    potion,
                    "potion",
                    f"potion:{slot['slot']}",
                    {
                        "content_id": cid,
                        "slot": slot["slot"],
                        "owner_ref": "player",
                        "target_type": potion.get("target_type"),
                        "stats": base_values(potion),
                        "effect_coverage": "opaque",
                        "semantic_program": opaque(cid),
                    },
                )
        # The map names the bosses of the act, in the order they are fought. Absent
        # from recordings made before the mod exported them: unknown.
        ends = sorted((n for n in state.get("map") or [] if n["type"] == "Boss"), key=lambda n: n["row"])
        bosses = dict(zip(map(self.map_ref, ends), (run.get("boss"), run.get("second_boss"))))
        for node in state.get("map") or []:
            ref = self.map_ref(node)
            current = run.get("location") or {}
            entity = {
                "entity_type": "map_node",
                "ref": ref,
                "content_id": node["type"],
                "floor": node["row"],
                "col": node["col"],
                "current": (node["row"], node["col"])
                == (current.get("row"), current.get("col")),
                "visited": any(self.map_ref(x) == ref for x in run.get("visited", [])),
            }
            if bosses.get(ref):
                entity["encounter"] = "ENCOUNTER." + bosses[ref]
            self.entities.append(entity)
            self.relations.extend(
                {"source": ref, "target": self.map_ref(child), "role": "map_edge"}
                for child in node.get("children", [])
            )
        self.context(state["legal"])
        self.memory = recorded_memory(state, self.refs, self.entities)
        self.relations.extend(draw_relations(self.memory))

    @staticmethod
    def map_ref(node):
        return f"map:{node['col']}:{node['row']}"

    def add(self, raw, kind, ref, fields):
        self.entities.append({"entity_type": kind, "ref": ref, **fields})
        if raw.get("instance_id") is not None:
            self.refs[raw["instance_id"]] = ref
        return ref

    def displayed(self, option, ref):
        """The numbers and names an option's visible text refers to, as the engine publishes them."""
        for name, amount in (option.get("displayed_variables") or {}).items():
            self.entities.append({"entity_type": "displayed_variable", "owner_ref": ref,
                                  "content_id": name, "amount": amount, "known": True})
        for name, content in (option.get("displayed_names") or {}).items():
            self.entities.append({"entity_type": "displayed_variable", "owner_ref": ref,
                                  "content_id": name, "names": content, "known": True})

    def powers(self, creature, ref):
        for power in creature.get("powers") or []:
            self.entities.append(
                {
                    "entity_type": "power",
                    "content_id": content_id(power, "power"),
                    "owner_ref": ref,
                    "stacks": power.get("amount"),
                    "stats": base_values(power),
                }
            )

    def creature(self, raw, kind, ref):
        self.add(
            raw,
            kind,
            ref,
            {
                "content_id": content_id(raw, "monster"),
                "known": True,
                **{k: raw.get(k) for k in ("hp", "max_hp", "block")},
            },
        )
        self.powers(raw, ref)
        for intent in raw.get("intents") or []:
            visible = {
                "entity_type": "intent",
                "owner_ref": ref,
                "intent": intent.get("type"),
            }
            if intent.get("damage") is not None:
                visible.update(
                    damage=intent["damage"],
                    hits=intent.get("repeats", 1),
                    total_damage=intent["damage"] * intent.get("repeats", 1),
                )
            self.entities.append(visible)

    def offer(self, cards, prefix):
        """Cards of a card reward, in the order shown. Unlike a pile, an offer has no summary."""
        return [
            self.add(card, "card", f"{prefix}:{i}",
                     {**card_fields(card), "zone": "reward", "owner_ref": "player"})
            for i, card in enumerate(cards)
        ]

    def pile(self, cards, zone, ordered=False, summary=True):
        # Canonicalize using public values only; instance IDs and hidden array order
        # never choose features, slots or positional encodings.
        cards = list(cards)
        if not ordered:
            cards.sort(key=lambda x: json.dumps(card_fields(x), sort_keys=True))
        for i, card in enumerate(cards):
            if card.get("instance_id") in self.refs:
                continue  # A deck card offered for removal is the same public object.
            ref = f"{zone}:{i}"
            self.add(
                card,
                "card",
                ref,
                {
                    **card_fields(card),
                    "zone": zone,
                    "owner_ref": "player",
                    **({"position": i} if ordered else {}),
                },
            )
        if not summary:
            return  # The cards offered by a selection are not a pile of their own.
        self.entities.append(
            {
                "entity_type": "pile_summary",
                "zone": zone,
                "count": len(cards),
                "complete": True,
                "order_known": ordered,
            }
        )

    def context(self, legal):
        scope, context = legal["scope"], legal.get("context")
        if scope == "event_choice":
            self.entities.append(
                {"entity_type": "event", "content_id": "EVENT." + context["event_id"]}
            )
            for i, option in enumerate(context["options"]):
                self.add(
                    option,
                    "event_option",
                    f"option:{i}",
                    {
                        "content_id": option["text_key"],
                        "enabled": not option.get("locked", False),
                        "effect_coverage": "opaque",
                        "semantic_program": opaque(option["text_key"]),
                    },
                )
                self.displayed(option, f"option:{i}")
        elif scope in {"rest_site", "treasure"}:
            for item in context:
                i = item["index"]
                if scope == "rest_site":
                    kind, cid = "rest_option", item["option_id"]
                else:
                    kind, cid = "relic", content_id(item["relic"], "relic")
                ref = f"option:{i}"
                self.add(
                    item,
                    kind,
                    ref,
                    {
                        "content_id": cid,
                        # A rest option can be shown disabled; a relic in an open chest is its zone.
                        **({"enabled": item.get("enabled", True)} if scope == "rest_site" else {"zone": "treasure"}),
                        "effect_coverage": "opaque",
                        "semantic_program": opaque(cid),
                    },
                )
                self.displayed(item, ref)
                self.keys[(scope, i)] = ref
        elif scope == "reward_choice":
            for i, item in enumerate(context["rewards"]):
                if item.get("selected"):
                    continue  # A collected reward has left the screen.
                cid = item["reward_type"] + "Reward"
                value = item.get("value")
                if isinstance(value, dict) and item["reward_type"] in {
                    "Potion",
                    "Relic",
                }:
                    cid = content_id(value, item["reward_type"])
                fields = {
                    "content_id": cid,
                    "effect_coverage": "opaque",
                    "semantic_program": opaque(cid),
                }
                if item["reward_type"] == "Gold":
                    fields["gold"] = value
                reward = self.add(item, "reward", f"reward:{i}", fields)
                # The cards of a card reward are generated before the screen is shown and
                # looking at them is free: the bridge publishes them with the reward.
                # Without that flag they are unknown until the separate selection opens.
                if not item.get("expanded_card_reward"):
                    continue
                for ref in self.offer(item["cards"], f"{reward}:card"):
                    self.relations.append({"source": reward, "target": ref, "role": "offers"})
                for option in item.get("alternatives") or []:
                    cid = option.get("OptionId")
                    require(isinstance(cid, str), "Unknown reward alternative")
                    ref = f"{reward}:alternative:{cid}"
                    self.entities.append(
                        {
                            "entity_type": "reward_alternative",
                            "ref": ref,
                            "content_id": cid,
                            "effect_coverage": "opaque",
                            "semantic_program": opaque(cid),
                        }
                    )
                    self.keys[(scope, item["instance_id"], cid)] = ref
        elif scope == "shop":
            for i, item in enumerate(context):
                value = item["value"]
                card = (value.get("CreationResult") or {}).get("Card")
                typename = value.get("type", "").rsplit(".", 1)[-1]
                kind = {"MerchantRelicEntry": "relic", "MerchantPotionEntry": "potion"}.get(typename)
                model = value.get("Model")
                # As the engine names an entry: by what it sells, or by its own type
                # when that is a service or nothing is left of it.
                cid = (content_id(card, "card") if card
                       else content_id(model, kind) if kind and model else typename)
                require(cid, "Missing visible shop item")
                ref = f"shop:{i}"
                if card:
                    offered = self.add(card, "card", f"{ref}:card",
                                       {**card_fields(card), "zone": "shop", "owner_ref": "player"})
                    self.relations.append({"source": ref, "target": offered, "role": "offers"})
                self.add(item, "shop_item", ref, {
                    "content_id": cid,
                    "price": value.get("Cost"),
                    "sold_out": not value.get("IsStocked", True),
                    "effect_coverage": "opaque",
                    "semantic_program": opaque(cid),
                })
        elif scope == "card_select":
            require(isinstance(context, list), "Missing pre-selection card offer")
            # Rebind offered cards to the option zone, not to an internal/deck order.
            self.pile(context, "selection", summary=False)
        elif scope == "card_reward":
            cards = context["cards"]
            for i, ref in enumerate(self.offer(cards, "reward")):
                self.keys[(scope, i)] = ref
            for i, option in enumerate(context["alternatives"], len(cards)):
                cid = option.get("OptionId")
                require(isinstance(cid, str), "Unknown reward alternative")
                ref = f"alternative:{i}"
                self.entities.append(
                    {
                        "entity_type": "reward_alternative",
                        "ref": ref,
                        "content_id": cid,
                        "effect_coverage": "opaque",
                        "semantic_program": opaque(cid),
                    }
                )
                self.keys[(scope, i)] = ref
        elif scope == "bundle_select":
            for bundle in context:
                ref = f"bundle:{bundle['index']}"
                self.entities.append(
                    {
                        "entity_type": "bundle",
                        "ref": ref,
                        "cards": [card_fields(c) for c in bundle["cards"]],
                    }
                )
                self.keys[(scope, bundle["index"])] = ref
        elif scope == "crystal_sphere":
            self.entities.append(
                {
                    "entity_type": "event_state",
                    "content_id": "CRYSTAL_SPHERE",
                    "remaining_steps": context["remaining"],
                }
            )
            for tool in ("Small", "Big"):
                self.entities.append(
                    {
                        "entity_type": "tool",
                        "ref": "tool:" + tool,
                        "content_id": tool,
                        "cost": 1,
                        "semantic_program": {
                            "kind": "effect",
                            "op": "REVEAL",
                            "shape": "cell" if tool == "Small" else "square",
                            "radius": 0 if tool == "Small" else 1,
                            "clip_to_board": True,
                            "count": 1,
                        },
                    }
                )
            for cell in context["cells"]:
                ref = f"cell:{cell['x']}:{cell['y']}"
                fields = {
                    "entity_type": "board_cell",
                    "ref": ref,
                    "x": cell["x"],
                    "y": cell["y"],
                    "revealed": not cell["hidden"],
                    "known": not cell["hidden"],
                }
                if not cell["hidden"]:
                    item = cell.get("item")
                    fields["content_id"] = (
                        item.get("type", "unknown").rsplit(".", 1)[-1]
                        if item
                        else "empty"
                    )
                self.entities.append(fields)
                self.keys[(scope, (cell["x"], cell["y"]))] = ref
        elif scope not in {"combat_play", "map_select"}:
            raise ProtocolError("Unsupported recorded decision scope: " + scope)

    def action(self, raw, index, scope):
        command, args = raw["command"], raw.get("args", {})
        require(command in VERBS, "Unsupported recorded command: " + command)
        result = {
            "candidate_ref": f"c{index}",
            "decoder_slot_ref": f"a{index}",
            "verb": VERBS[command],
        }
        if (
            command in {"pick_relic", "select_reward_option", "select_bundle"}
            and args.get("index") is None
        ):
            result["verb"] = "LEAVE_ROOM" if command == "pick_relic" else "SKIP"
        elif command == "take_reward_alternative":
            result["source_ref"] = self.keys[(scope, args["reward_instance_id"], args["option_id"])]
        else:
            source = next(
                (
                    args[k]
                    for k in (
                        "card_instance_id",
                        "potion_instance_id",
                        "option_instance_id",
                        "reward_instance_id",
                        "entry_instance_id",
                    )
                    if k in args
                ),
                None,
            )
            if source is not None:
                require(source in self.refs, "Candidate refers to an unobserved source")
                result["source_ref"] = self.refs[source]
            elif command == "select_map_node":
                result["source_ref"] = self.map_ref(args)
                require(
                    any(e.get("ref") == result["source_ref"] for e in self.entities),
                    "Missing visible map node",
                )
            elif command in {
                "choose_rest_option",
                "pick_relic",
                "select_reward_option",
                "select_bundle",
            }:
                result["source_ref"] = self.keys[(scope, args["index"])]
                if command == "select_reward_option" and result[
                    "source_ref"
                ].startswith("alternative:"):
                    result["verb"] = "CHOOSE_REWARD_ALTERNATIVE"
            elif command == "crystal_sphere_cell":
                require(args["tool"] in {"Small", "Big"}, "Unknown crystal sphere tool")
                result["source_ref"] = "tool:" + args["tool"]
                result["target_refs"] = [self.keys[(scope, (args["x"], args["y"]))]]
        target = args.get("target_instance_id")
        if target is not None:
            require(target in self.refs, "Candidate refers to an unobserved target")
            result["target_refs"] = [self.refs[target]]
        # The shape of an engine candidate.
        source = result.pop("source_ref", None)
        result["source_refs"] = [] if source is None else [source]
        result.setdefault("target_refs", [])
        result["effect_coverage"] = "opaque"
        return result
