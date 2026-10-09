"""Public rule text as local programs.

The game states what a card, relic, potion, power or event option does in its
description, and the player reads exactly that. The engine marks these rules
opaque, so the description template is rendered with the entity's public
values and tokenized into a program: content which shares words shares
structure, instead of being known only by its identifier.
"""

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from .representation import Effect, fields_of

LOCALIZATION = Path(__file__).resolve().parents[1] / "sts2-cli" / "localization_eng"
# Longest descriptions are far shorter; the limit only bounds a malformed entry.
TOKEN_LIMIT = 128

# Category prefix of a content id -> (table, description keys in order of preference).
CATEGORIES = {
    "CARD": ("cards", ("description",)),
    "RELIC": ("relics", ("description",)),
    "POTION": ("potions", ("description",)),
    "POWER": ("powers", ("smartDescription", "description")),
    "ORB": ("orbs", ("smartDescription", "description")),
    "ENCHANTMENT": ("enchantments", ("extraCardText", "description")),
    "AFFLICTION": ("afflictions", ("extraCardText", "description")),
}
TABLES = sorted({table for table, _ in CATEGORIES.values()} | {"events", "ancients", "rest_site_ui"})
ICONS = {"energyIcons": "energy", "starIcons": "star"}
KEYWORD_FORMATS = ("plural", "cond", "show")
# Formatters which display a value other than the stored one.
DISPLAY = {
    "abs": abs,
    "inverseDiff": lambda value: -value,
    "percentMore": lambda value: round((value - 1) * 100, 4),
    "percentLess": lambda value: round((1 - value) * 100, 4),
}
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[.,:%+]")
MARKUP = re.compile(r"\[/?[A-Za-z_]+(?:[= ][^\]]*)?\]")
CALL = re.compile(r"([A-Za-z]+)\(([^()]*)\)(?::|$)")
CONDITION = re.compile(r"\s*(<=|>=|!=|<|>|=)\s*(-?\d+(?:\.\d+)?)\?")
VARIABLE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)")


@lru_cache(maxsize=None)
def _tables():
    if not LOCALIZATION.is_dir():
        raise FileNotFoundError(f"Rule text needs the localization tables in {LOCALIZATION}")
    return {name: json.loads((LOCALIZATION / f"{name}.json").read_text()) for name in TABLES}


@lru_cache(maxsize=None)
def digest():
    """Identity of the rule text a checkpoint was trained with."""
    return hashlib.sha256(
        json.dumps(_tables(), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def reward_alternatives():
    """Option ids of what a card reward offers beside its cards: reroll, sacrifice, skip."""
    table = json.loads((LOCALIZATION / "card_reward_ui.json").read_text())
    return {key[7:-5] for key in table if key.startswith("OPTION_") and key.endswith(".name")}


@lru_cache(maxsize=None)
def canonical_content_id(content_id):
    """Dynamic relic event options use the relic's title key, not a new content."""
    if content_id.endswith(".title"):
        entry = content_id[:-6]
        matches = [name for name, table in _tables().items() if content_id in table]
        if matches == ["relics"] and f"{entry}.description" in _tables()["relics"]:
            return "RELIC." + entry
    return content_id


@lru_cache(maxsize=None)
def template(content_id):
    """Description template of a content id, or None when the game states none."""
    tables = _tables()
    content_id = canonical_content_id(content_id)
    category, _, entry = content_id.partition(".")
    if category in CATEGORIES:
        table, keys = CATEGORIES[category]
        return next((tables[table][f"{entry}.{key}"] for key in keys if f"{entry}.{key}" in tables[table]), None)
    for table, key in (("events", content_id), ("ancients", content_id), ("rest_site_ui", "OPTION_" + content_id)):
        if f"{key}.description" in tables[table]:
            return tables[table][f"{key}.description"]
    if ".options." in content_id:
        # Ancient options offer a relic and take their text from it.
        relic = content_id.rsplit(".", 1)[1]
        for key in ("eventDescription", "description"):
            if f"{relic}.{key}" in tables["relics"]:
                return tables["relics"][f"{relic}.{key}"]
    return None


@lru_cache(maxsize=None)
def variables(content_id):
    """Names of the values a description displays; these are public with it."""
    text = template(content_id) if isinstance(content_id, str) else None
    return frozenset(VARIABLE.findall(text)) if text else frozenset()


def _closing(text, start):
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Unbalanced rule text")


def _options(text):
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        depth += (char == "{") - (char == "}")
        if char == "|" and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    return parts + [text[start:]]


def _number(value):
    return value if type(value) in (int, float) else None


def _words(text):
    for match in WORD.finditer(MARKUP.sub(" ", text)):
        word = match.group()
        yield ("number", float(word) if "." in word else int(word)) if word[0].isdigit() else ("word", word.lower())


def _render(text, values, current=None):
    """Tokens of a template: words, literal numbers and displayed variables."""
    tokens, position = [], 0
    while True:
        start = text.find("{", position)
        if start < 0:
            tokens.extend(_words(text[position:]))
            return tokens
        tokens.extend(_words(text[position:start]))
        end = _closing(text, start)
        tokens.extend(_placeholder(text[start + 1 : end], values, current))
        position = end + 1


def _placeholder(body, values, current):
    name, _, rest = body.partition(":")
    variable = (name, values.get(name.lower())) if name else current
    if variable is None:
        return []
    value = variable[1]
    call = CALL.match(rest)
    if call:
        formatter, argument, options = call.group(1), call.group(2), rest[call.end() :]
    else:
        formatter = next((k for k in KEYWORD_FORMATS if rest.startswith(k + ":")), None)
        argument, options = "", rest[len(formatter) + 1 :] if formatter else rest
    if not options and formatter != "choose":
        if not rest or call:
            literal = formatter in ICONS and argument.strip().isdigit()
            number = _number(value)
            if number is not None and formatter in DISPLAY:
                variable = (variable[0], DISPLAY[formatter](number))
            shown = [("number", int(argument))] if literal else [("variable", *variable)]
            return shown + ([("word", ICONS[formatter])] if formatter in ICONS else [])
    choices = _options(options)
    if formatter == "plural":
        number = _number(value)
        chosen = choices[-1] if number is None or number != 1 or len(choices) == 1 else choices[0]
    elif formatter == "cond":
        chosen, number = CONDITION.sub("", choices[-1]), _number(value)
        for choice in choices if number is not None else ():
            condition = CONDITION.match(choice)
            if condition is None:
                chosen = choice
                break
            operator, bound = condition.group(1), float(condition.group(2))
            if {"<": number < bound, "<=": number <= bound, ">": number > bound,
                    ">=": number >= bound, "=": number == bound, "!=": number != bound}[operator]:
                chosen = choice[condition.end() :]
                break
    elif formatter == "choose":
        names = argument.split("|")
        chosen = choices[names.index(str(value))] if str(value) in names else choices[-1]
    else:
        # A flag selects between its two texts; an unknown flag reads as unset.
        chosen = choices[0] if value is True else choices[1] if len(choices) > 1 else ""
    return _render(chosen, values, variable)


def _shown(value):
    """A stat as displayed: its number, or None while it is masked as unknown."""
    if not isinstance(value, dict):
        return value
    masked = value.get("known") is False or value.get("applicable") is False
    return None if masked else value.get("value")


def _values(entity):
    values = {}
    def public(name):
        if ((entity.get("known_masks") or {}).get(name) is False
                or (entity.get("applicable_masks") or {}).get(name) is False):
            return None
        return _shown(entity.get(name))
    for key, value in (entity.get("stats") or {}).items():
        values[key.lower()] = _shown(value)
    # A name the text displays stands for a content, given by its id.
    for key, value in (entity.get("named") or {}).items():
        values[key.lower()] = value
    values["ifupgraded"] = entity.get("upgraded") is True
    values["ismultiplayer"] = False
    if public("card_type") is not None:
        values["cardtype"] = public("card_type")
    if entity.get("content_id") == "CARD.MAD_SCIENCE":
        rider = public("rider_effect")
        values["hasrider"] = rider != "None" if isinstance(rider, str) else None
        for name in ("Violence", "Sapping", "Choking", "Energized", "Wisdom", "Chaos",
                     "Expertise", "Curious", "Improvement"):
            values[name.lower()] = rider == name if isinstance(rider, str) else None
    if entity.get("entity_type") == "power":
        values["onplayer"] = entity.get("owner_ref") == "player"
    amount = entity.get("stacks", entity.get("counter"))
    if amount is not None:
        values.setdefault("amount", amount)
    return values


def program(entity):
    """The rule text of an entity as a program, or None when it has none."""
    content = entity.get("content_id")
    if not isinstance(content, str) or template(content) is None:
        return None
    parts = [(content, _values(entity))]
    for key in ("enchantment", "affliction"):
        if isinstance(entity.get(key), str) and template(entity[key]) is not None:
            # An enchantment displays one amount, under whichever name its text uses.
            amount = entity.get(key + "_amount")
            parts.append((entity[key], {name.lower(): amount for name in variables(entity[key])}))
    key = tuple((content, tuple(sorted(values.items()))) for content, values in parts)
    try:
        return _program(key)
    except TypeError:
        # A value that cannot be hashed is rendered without the cache.
        return _program.__wrapped__(key)


@lru_cache(maxsize=16384)
def _program(parts):
    tokens = []
    for content, values in parts:
        tokens.extend(_render(template(content), dict(values)))
    tokens = tokens[:TOKEN_LIMIT]
    if not tokens:
        return None
    effect = Effect(nodes=[fields_of({"kind": "rule_text"})])
    for kind, *token in tokens:
        if kind == "variable":
            name, value = token
            if isinstance(value, str):
                # A displayed name: the content it stands for.
                node = {"variable": name, "content_id": value}
            else:
                # A displayed variable whose value is not public stays an explicit unknown.
                node = {"variable": name, "value": {"value": _number(value), "known": _number(value) is not None}}
        else:
            node = {kind: token[0]}
        effect.nodes.append(fields_of(node))
    count = len(effect.nodes)
    for i in range(1, count):
        effect.edges += [(0, i, "text_contains"), (i, 0, "text_within")]
        for j in range(i + 1, count):
            near = {1: "", 2: "2"}.get(j - i)
            if near is None:
                effect.edges += [(i, j, "text_after"), (j, i, "text_before")]
            else:
                effect.edges += [(i, j, "text_next" + near), (j, i, "text_previous" + near)]
    return effect


def _templates():
    for name, table in _tables().items():
        for key, text in table.items():
            if name in ("events", "ancients"):
                used = ".options." in key and key.endswith(".description")
            elif name == "rest_site_ui":
                used = key.startswith("OPTION_") and key.endswith(".description")
            else:
                used = key.rsplit(".", 1)[-1] in ("description", "smartDescription", "extraCardText", "eventDescription")
            if used:
                yield text


@lru_cache(maxsize=None)
def displayed():
    """Every variable name any rule text displays."""
    return frozenset(name for text in _templates() for name in VARIABLE.findall(text))


def symbols():
    """Every word and variable any rule text can produce, whatever its values."""
    result = {"variable=" + name for name in displayed()}
    for text in _templates():
        for kind, word in _words(re.sub(r"[{}|()?<>=]", " ", text)):
            if kind == "word":
                result.add("word=" + word)
    return result | {"word=" + word for word in ICONS.values()} | {"kind=rule_text", "number=<number>", "value=<number>"}


TEXT_RELATIONS = ("text_contains", "text_within", "text_after", "text_before",
                  "text_next", "text_previous", "text_next2", "text_previous2")
TEXT_FIELDS = ("kind", "word", "number", "variable", "value", "content_id")
