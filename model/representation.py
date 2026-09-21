"""Typed public fields, local effect programs and reference-only entity handles.

Packing indices never become features. Piles remain full unordered multisets.
Symbols are a frozen, checkpointed vocabulary; unknown content is explicit.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math

import torch

from .protocol import ProtocolError, fingerprint

ENTITY_FIELDS = set("entity_type content_id zone character phase act floor hp max_hp block gold energy max_energy stars turn round cost base_cost star_cost base_star_cost x_cost upgraded upgrade_level card_type rarity enchantment affliction stacks count capacity slot position known applicable exhausted used remaining charges target_type intent damage hits total_damage current visited selectable complete revealed operation destination scope effect_coverage quantity max_slots order_known valid_until event_type stage mode min_total max_total selected_count remaining_required remaining_steps can_finish can_skip can_cancel order_matters repetition_allowed nonboss_paid boss_1 boss_2 boss_3".split())
ENTITY_FIELDS.update("x y col row price sold_out rarity enchantment_amount affliction_amount visibility remaining_uses step_index elapsed_turns trigger_period remaining_turns incoming_damage nominal_block_margin energy_margin gold_margin stars_margin".split())
ENTITY_FIELDS.update("enabled base_star_cost current_star_cost source amount".split())
ENTITY_FIELDS.update("rider_effect retain stars_x exhaust_on_next_play".split())
CONTAINERS = set("stats keywords modifiers powers intents public_previews public_costs known_masks applicable_masks after_upgrade fields properties numeric".split())
STAT_FIELDS = set("damage block draw cards energy stars hp maxhp heal strength dexterity vulnerable weak frail poison exhaust discard summon repeat times amount count turns cost max_select min_select".split())
STAT_FIELDS.update({"passive", "evoke"})
# Mad Science declares all of these public parameters on every variant. The
# separate rider_effect and card_type identify which ones the card uses.
STAT_FIELDS.update("sappingweak sappingvulnerable violencehits chokingdamage energizedenergy wisdomcards expertisestrength expertisedexterity curiousreduction".split())
REFERENCE_FIELDS = {"ref", "source_ref", "owner_ref", "source_refs", "target_refs", "option_refs", "selected_refs"}
ACTION_FIELDS = ENTITY_FIELDS | CONTAINERS | REFERENCE_FIELDS | {"verb", "decoder_slot_ref", "candidate_ref", "semantic_program", "program"}
PROGRAM_FIELDS = set("kind op args bindings role timing reason children steps body condition then else cases repeat probability count amount value known applicable unit content_id object_kind field read_at domain distribution alternatives weights min max ordered order_matters replacement options choice trigger duration left right operand recipient subject actor source target destination cost consequence selector scope name entity ref program on_use_program definition_id".split())
PROGRAM_FIELDS.update("acquisition_method bind board bound_roles cancelable cause center clip_to_board collection completion completion_rules context context_ref continuation controller coverage damage_flags decision_kind definition_ref delta enchantment evaluate_at event filters generator groups identity insufficient_rule iteration limit mode operation owner per_event predicate prospective_effect public_generator_rule public_preview public_rule radius rarity requested_count resource result_identity reveal_at reveal_semantics_ref selected_member selected_refs selection_effect_ref shape stacks target_type times usage uses_cost refs".split())
BINDING_ROLES = set("actor recipient subject source target owner destination option payer beneficiary card potion relic enemy player self other".split())


def _program(value):
    if isinstance(value, list):
        return [_program(x) for x in value]
    if not isinstance(value, dict):
        return value
    result = {k: _program(v) for k, v in value.items() if k in PROGRAM_FIELDS or k in BINDING_ROLES}
    if result.get("known") is False:
        result = {k: v for k, v in result.items() if k in {"kind", "known", "applicable", "unit"}}
    return result


def clean_entity(value, *, action=False):
    allowed = ACTION_FIELDS if action else ENTITY_FIELDS | CONTAINERS | REFERENCE_FIELDS | {"semantic_program", "program", "cards"}
    result = {}
    for key, item in value.items():
        if key not in allowed:
            continue
        if key in {"semantic_program", "program"}:
            result[key] = _program(item)
        elif key == "cards":
            result[key] = [clean_entity(x) for x in item]
        elif key == "stats":
            stats = {k: _public_value(v) for k, v in item.items() if k.lower().replace("_", "") in STAT_FIELDS}
            if stats:
                result[key] = stats
        elif key in CONTAINERS and isinstance(item, dict):
            result[key] = {k: _public_value(v) for k, v in item.items()
                           if k in ENTITY_FIELDS or k.lower() in STAT_FIELDS or k in BINDING_ROLES}
        elif key in CONTAINERS and isinstance(item, list):
            result[key] = [clean_entity(x) if isinstance(x, dict) else x for x in item]
        else:
            result[key] = _public_value(item)
    if result.get("zone") in {"draw_pile", "discard_pile", "exhaust_pile"}:
        # Publicly revealed order has a separate relation, never a raw array index.
        for key in ("position", "slot", "row", "col"):
            result.pop(key, None)
    if result.get("revealed") is False or result.get("known") is False:
        public_shape = {"ref", "entity_type", "zone", "x", "y", "row", "col", "position", "count", "capacity", "known", "applicable", "revealed"}
        result = {k: v for k, v in result.items() if k in public_shape}
    return result


def _public_value(item):
    if isinstance(item, dict):
        # Numeric fields may carry known/applicable masks alongside their value.
        return {k: _public_value(v) for k, v in item.items()
                if k in ENTITY_FIELDS | {"value", "unit"}}
    if isinstance(item, list):
        return [_public_value(x) for x in item]
    return item


def clean_public(public):
    result = {"phase": public.get("phase", "unknown"),
              "entities": [clean_entity(x) for x in public.get("entities", [])],
              "memory": [clean_entity(x) for x in public.get("memory", [])],
              "relations": [{k: v for k, v in x.items() if k in {"source", "target", "role"}}
                            for x in public.get("relations", [])]}
    for key in ("selection_context", "event_context"):
        if public.get(key):
            result[key] = clean_entity(public[key])
    if public.get("decoder_bank") is not None:
        result["decoder_bank"] = [clean_entity(x, action=True) for x in public["decoder_bank"]]
    return result


def bucket(text, size):
    return 1 + int.from_bytes(hashlib.blake2s(text.encode(), digest_size=4).digest(), "big") % (size - 1)


class Vocabulary:
    def __init__(self, symbols=None, capacity=16384):
        self.symbols = ["<pad>", "<unknown>"] + sorted(set(symbols or []) - {"<pad>", "<unknown>"})
        if len(self.symbols) > capacity:
            raise ValueError("Vocabulary exceeds configured capacity; increase capacity explicitly")
        self.capacity = capacity
        self.lookup = {s: i for i, s in enumerate(self.symbols)}

    def encode(self, symbol):
        return self.lookup.get(symbol, 1)

    @property
    def digest(self):
        return fingerprint(self.symbols)


@dataclass
class Field:
    name: str
    symbol: str
    number: tuple[float, float, float, float, float]


def fields_of(value, prefix=""):
    fields = []
    known = value.get("known_masks", {}) if isinstance(value, dict) else {}
    applicable = value.get("applicable_masks", {}) if isinstance(value, dict) else {}
    if isinstance(value, dict):
        for key in sorted(value):
            if key in REFERENCE_FIELDS | {"decoder_slot_ref", "candidate_ref", "semantic_program", "program", "cards", "known_masks", "applicable_masks"}:
                continue
            item = value[key]
            name = f"{prefix}.{key}" if prefix else key
            if known.get(key) is False:
                item = {"known": False, "applicable": applicable.get(key, True)}
            if applicable.get(key) is False:
                item = {"known": False, "applicable": False}
            if isinstance(item, dict) and ("value" in item or item.get("known") is False or item.get("applicable") is False):
                fields.append(scalar_field(name, item.get("value"), item.get("known", True), item.get("applicable", True)))
            elif isinstance(item, (dict, list)):
                fields.extend(fields_of(item, name))
            else:
                fields.append(scalar_field(name, item))
        for key in sorted((known.keys() | applicable.keys()) - value.keys()):
            # Unknown references must differ from roles which do not apply.
            name = f"{prefix}.{key}" if prefix else key
            fields.append(scalar_field(name, True if known.get(key) else None,
                                       known.get(key, False), applicable.get(key, True)))
    elif isinstance(value, list):
        for item in value:  # Unordered repeated fields retain multiplicity, no index embedding.
            fields.extend(fields_of(item, prefix) if isinstance(item, (dict, list)) else [scalar_field(prefix, item)])
    return fields or [scalar_field(prefix or "empty", None)]


def scalar_field(name, value, known=True, applicable=True):
    known = bool(known and value is not None and applicable)
    if not known:
        return Field(name, "<unknown>", (0, 0, 0, float(applicable), 0))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ProtocolError("Non-finite public number")
        scale = 100.0 if any(x in name for x in ("hp", "gold", "damage", "block")) else 10.0
        return Field(name, name + "=<number>", (value / scale, math.copysign(math.log1p(abs(value)), value), 1, 1, 1))
    return Field(name, f"{name}={value}", (0, 0, 1, 1, 0))


@dataclass
class Effect:
    nodes: list[list[Field]] = field(default_factory=list)
    edges: list[tuple[int, int, str]] = field(default_factory=list)
    bindings: list[tuple[int, str, str]] = field(default_factory=list)


def effect_tree(program):
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
        own = {k: v for k, v in obj.items() if not isinstance(v, (dict, list)) and k not in {"ref", "bind"}
               and not (k == "name" and obj.get("kind") == "variable")}
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
                effect.edges.extend([(binder, index, "binds_variable"), (index, binder, "variable_from")])
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


@dataclass
class Observation:
    tokens: list[list[Field]]
    effects: list[Effect]
    refs: dict[str, int]
    action_indices: list[int]
    slot_refs: list[str]
    edges: list[tuple[int, int, str]]
    map_floors: dict[int, int]
    context: dict
    digest: str


def observation(frame):
    public = clean_public(frame["public"])
    entities = [{"entity_type": "global", "phase": public["phase"]}]
    entities += public["entities"] + public["memory"]
    if public.get("event_context"):
        entities.append(dict(public["event_context"], entity_type="event_context"))
    edges = []
    # Bundle contents stay separate entities, linked to their containing bundle.
    for item in list(entities):
        for i, card in enumerate(item.pop("cards", [])):
            ref = f"bundle-child:{len(entities)}:{i}"
            entities.append(dict(card, ref=ref, owner_ref=item["ref"]))
    bank = public.get("decoder_bank") or [clean_entity(c, action=True) for c in frame["legal"]["candidates"]]
    action_start = len(entities)
    entities += [dict(c, entity_type="action") for c in bank]
    refs = {}
    for i, item in enumerate(entities):
        ref = item.get("ref")
        if ref:
            if ref in refs:
                raise ProtocolError("Duplicate public entity reference")
            refs[ref] = i
    for i, item in enumerate(entities):
        for role in ("source", "target", "option", "owner"):
            bindings = item.get(role + "_refs", [])
            if item.get(role + "_ref"):
                bindings = bindings + [item[role + "_ref"]]
            for ref in bindings:
                if ref not in refs:
                    raise ProtocolError(f"Unresolved public {role} reference")
                edges += [(i, refs[ref], role), (refs[ref], i, "reverse_" + role)]
    for edge in public["relations"]:
        if edge.get("source") not in refs or edge.get("target") not in refs:
            raise ProtocolError("Unresolved relation")
        edges.append((refs[edge["source"]], refs[edge["target"]], edge["role"]))
    floors = {i: int(x.get("floor", 0)) for i, x in enumerate(entities) if x.get("entity_type") == "map_node"}
    edges += map_relations(floors, edges)
    context = public.get("selection_context") or {}
    # Dynamic prefix/mask never contaminates cached Transformer input.
    digest = fingerprint({"entities": entities, "edges": edges})
    return Observation([fields_of(x) for x in entities],
                       [effect_tree(x.get("semantic_program") or x.get("program")) for x in entities],
                       refs, list(range(action_start, len(entities))),
                       [c["decoder_slot_ref"] for c in bank], edges, floors, context, digest)


def map_relations(floors, edges):
    direct = {(i, j) for i, j, role in edges if role == "map_edge"}
    adjacency = {i: {j for a, j in direct if a == i} for i in floors}
    reachable = {}

    def reach(i, stack):
        if i in stack:
            raise ProtocolError("Map must be a DAG")
        if i not in reachable:
            reachable[i] = set()
            for j in adjacency.get(i, ()):
                reachable[i].add(j)
                reachable[i].update(reach(j, stack | {i}))
        return reachable[i]

    for i in floors:
        reach(i, set())
    result = []
    for i in floors:
        for j in floors:
            role = ("self" if i == j else "forward_direct" if (i, j) in direct else
                    "reverse_direct" if (j, i) in direct else "forward_indirect" if j in reachable[i] else
                    "reverse_indirect" if i in reachable[j] else "unreachable")
            result.append((i, j, "map_" + role))
    return result


def symbols_from_frames(frames):
    # Schema categories are public constants. Initializing from only a Neow
    # snapshot must not collapse combat/map/shop phases to the same unknown ID.
    categories = {
        "phase": "combat map event rest_site shop treasure rewards card_reward card_select bundle_select crystal_sphere",
        "entity_type": "global action player card pile_summary enemy summon intent orb_slots orb power relic potion map_node displayed_variable event event_option rest_option shop_item reward reward_alternative bundle event_state tool board_cell reward_progress event_context",
        "zone": "deck hand draw_pile discard_pile exhaust_pile selection reward treasure shop",
        "operation": "unknown transform exhaust remove enchant discard upgrade copy obtain",
        "source": "unknown hand deck draw_pile discard_pile exhaust_pile",
        "destination": "unknown hand deck draw_pile discard_pile exhaust_pile removed",
        "mode": "buffered sequential Small Big",
        "effect_coverage": "opaque partial complete unknown",
        "character": "IRONCLAD SILENT DEFECT REGENT NECROBINDER Ironclad Silent Defect Regent Necrobinder",
        "card_type": "None Attack Skill Power Status Curse Quest",
        "rarity": "None Basic Common Uncommon Rare Ancient Special Token Event Status Curse Quest",
        "target_type": "None Self AnyEnemy AllEnemies RandomEnemy AnyPlayer AnyAlly AllAllies TargetedNoCreature Osty",
        "intent": "Attack Buff Debuff DebuffStrong Defend Escape Heal Hidden Sleep StatusCard CardDebuff DeathBlow Stun Summon Unknown",
        "content_id": "Small Big CrystalGold CrystalRelic CrystalPotion CrystalCurse CrystalCard CRYSTAL_SPHERE",
    }
    symbols = {f"{name}={value}" for name, values in categories.items() for value in values.split()}
    for name in "known applicable x_cost upgraded exhausted used current visited selectable complete revealed order_known enabled sold_out can_finish can_skip can_cancel order_matters repetition_allowed boss_1 boss_2 boss_3".split():
        symbols.update({f"{name}=True", f"{name}=False"})
    for frame in frames:
        obs = observation(frame)
        rows = obs.tokens + [n for e in obs.effects for n in e.nodes] + [fields_of(obs.context)]
        for row in rows:
            symbols.update(f.symbol for f in row)
    return symbols


def field_tensors(rows, vocabulary, field_buckets, device, length=None):
    n = length or len(rows)
    width = max(map(len, rows), default=1)
    ids = torch.zeros((n, width), dtype=torch.long, device=device)
    kinds = torch.zeros_like(ids)
    nums = torch.zeros((n, width, 5), device=device)
    mask = torch.zeros((n, width), dtype=torch.bool, device=device)
    # Construct on CPU in one pass; transfer once instead of tiny CUDA assignments.
    raw_ids, raw_kinds, raw_nums, raw_mask = [], [], [], []
    for row in rows + [[]] * (n - len(rows)):
        pad = width - len(row)
        raw_ids.append([vocabulary.encode(f.symbol) for f in row] + [0] * pad)
        raw_kinds.append([bucket(f.name, field_buckets) for f in row] + [0] * pad)
        raw_nums.append([f.number for f in row] + [(0,) * 5] * pad)
        raw_mask.append([True] * len(row) + [False] * pad)
    if raw_ids:
        ids = torch.tensor(raw_ids, dtype=torch.long, device=device)
        kinds = torch.tensor(raw_kinds, dtype=torch.long, device=device)
        nums = torch.tensor(raw_nums, dtype=torch.float32, device=device)
        mask = torch.tensor(raw_mask, dtype=torch.bool, device=device)
    return ids, kinds, nums, mask
