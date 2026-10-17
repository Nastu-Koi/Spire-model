"""A journal's decisions as macro-actions: one ordinary decision (`decision_macro`) or a
whole selection (`selection_macro`), with the cancel rules the recorder and the live
control share."""

from main.encoding.frames import clean_frame
from main.env.protocol import ProtocolError, fingerprint
from main.demos.recorder.snapshot import PublicSnapshot, require


PHASES = {
    "combat_play": "combat",
    "map_select": "map",
    "event_choice": "event",
    "rest_site": "rest_site",
    "shop": "shop",
    "treasure": "treasure",
    "reward_choice": "rewards",
    "card_select": "card_select",
    "card_reward": "card_reward",
    "bundle_select": "bundle_select",
    "crystal_sphere": "crystal_sphere",
}


SELECTION_CANCEL = "combat-or-dead-end-v1"


NONCOMBAT_CANCEL = "Non-combat selection cancel is an undo, not a label"


CANCEL_OPENERS = {"rest_site", "shop", "rewards"}


def cancel_offered(state, cancelable, can_finish, can_select):
    """Outside combat a native cancel only restores the state before the
    action that opened the selection, so it is offered only when nothing else is."""
    in_combat = bool((state.get("combat") or {}).get("in_progress"))
    return bool(cancelable) and (in_combat or (not can_finish and not can_select))


def decision_macro(decision, run_id, contract):
    require(decision.get("status") == "completed", "Decision was not completed")
    require(
        decision.get("state_boundary")
        in {"execution_or_semantic_entry", "input_submission", "selection_offer"},
        "Unknown pre-decision boundary",
    )
    state = decision["state"]
    require(state["run"].get("ascension") == contract["fixed_ascension"], "Decision ascension differs from the run")
    legal = state.get("legal") or {}
    require(
        legal.get("schema") == 1 and legal.get("status") == "complete",
        "Missing complete recorded legal set",
    )
    actions = legal.get("actions", [])
    require(
        actions and len({fingerprint(x) for x in actions}) == len(actions),
        "Empty or duplicate recorded legal actions",
    )
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
        require(
            selection
            and selection.get("min") == selection.get("max") == 1
            and len(ids) == 1,
            "Selection requires a recorded prefix path; final set alone is insufficient",
        )
        require(
            not cancel_offered(state, selection.get("cancelable"), False, True),
            "Cancelable selection lacks explicit cancel candidates",
        )
        choice = {"command": "select_card", "args": {"card_instance_id": ids[0]}}
    matches = [i for i, action in enumerate(actions) if action == choice]
    require(
        len(matches) == 1, "Recorded choice is not uniquely in the complete legal set"
    )
    require(
        decision.get("legal_match", {}).get("status") == "matched",
        "Recorder did not verify choice legality",
    )
    public = {
        "phase": PHASES[scope],
        "entities": snapshot.entities,
        "relations": snapshot.relations,
        "memory": snapshot.memory,
        "decoder_bank": bank,
    }
    frame = {
        "type": "decision_frame",
        "contract": contract,
        "boundary": "decision",
        "routing": {
            "episode_id": run_id,
            "decision_id": decision["action_id"],
            "state_version": decision["order"],
        },
        "public": public,
        "legal": {"candidates": bank},
        "events": [],
    }
    return {
        "phase": PHASES[scope],
        "steps": [
            {
                "frame": clean_frame(frame),
                "candidate_ref": bank[matches[0]]["candidate_ref"],
                "forced": len(bank) == 1,
            }
        ],
    }


def selection_macro(decision, run_id, contract, snapshot):
    """Consume recorder-emitted prefix masks; verify them against the frozen offer."""
    legal = decision["state"]["legal"]
    path, metadata = decision["selection_path"], legal["selection"]
    require(
        path.get("mode") == "buffered"
        and path.get("reconstructed") is True
        and path.get("order_source") == "submitted_result",
        "Unknown selection path semantics",
    )
    require(
        decision["choice_key"]["command"] == "select_cards",
        "Selection path label is not a card set",
    )
    require(
        decision.get("legal_match", {}).get("status") == "matched",
        "Invalid recorded selection",
    )
    offered = legal["actions"]
    require(
        all(a["command"] == "select_card" for a in offered),
        "Unexpected action in selection offer",
    )
    ids = [a["args"]["card_instance_id"] for a in offered]
    result = decision["choice_key"]["args"]["card_instance_ids"]
    minimum, maximum, cancelable = (
        metadata["min"],
        metadata["max"],
        metadata["cancelable"],
    )
    require(
        type(minimum) is int
        and type(maximum) is int
        and 0 <= minimum <= maximum
        and type(cancelable) is bool,
        "Invalid selection bounds",
    )
    require(
        len(result) == len(set(result))
        and set(result) <= set(ids)
        and len(result) <= maximum
        and (len(result) >= minimum or (not result and cancelable)),
        "Invalid submitted selection",
    )
    finish, cancel = (
        {"command": "finish_selection", "args": {}},
        {"command": "cancel", "args": {}},
    )
    require(len(path["steps"]) == len(result) + 1, "Incomplete selection prefix path")
    state = decision["state"]
    offers = [
        cancel_offered(state, cancelable, revision >= minimum,
                       revision < maximum and len(ids) > revision)
        for revision in range(len(path["steps"]))
    ]
    if len(result) < minimum and not offers[len(result)]:
        raise ProtocolError(NONCOMBAT_CANCEL)
    raw_bank = offered + [finish] + ([cancel] if any(offers) else [])
    bank = [snapshot.action(a, i, "card_select") for i, a in enumerate(raw_bank)]
    steps = []
    for revision, part in enumerate(path["steps"]):
        prefix = result[:revision]
        require(
            part["selected_ids"] == prefix,
            "Selection prefix disagrees with submitted order",
        )
        expected = (
            [a for a in offered if a["args"]["card_instance_id"] not in prefix]
            if len(prefix) < maximum
            else []
        )
        if len(prefix) >= minimum:
            expected += [finish]
        # The recorder masks the native UI, which always shows cancel when cancelable.
        require(
            part["actions"] == expected + ([cancel] if cancelable else []),
            "Incomplete or invented selection prefix mask",
        )
        if offers[revision]:
            expected += [cancel]
        label = (
            {"command": "select_card", "args": {"card_instance_id": result[revision]}}
            if revision < len(result)
            else cancel
            if len(result) < minimum
            else finish
        )
        require(
            part["choice_key"] == label and label in expected,
            "Selection prefix label is illegal",
        )
        candidates = [bank[raw_bank.index(a)] for a in expected]
        context = {
            "mode": "buffered",
            "min_total": minimum,
            "max_total": maximum,
            "selected_refs": [snapshot.refs[x] for x in prefix],
            "selected_count": len(prefix),
            "remaining_required": max(0, minimum - len(prefix)),
            "can_finish": finish in expected,
            "can_cancel": cancel in expected,
            "can_skip": False,
            "repetition_allowed": False,
            "order_matters": metadata.get("ordered", True),
        }
        frame = {
            "type": "decision_frame",
            "contract": contract,
            "boundary": "decision",
            "routing": {
                "episode_id": run_id,
                "decision_id": f"{decision['action_id']}:{revision}",
                "state_version": revision,
                "selection_id": decision["action_id"],
                "selection_revision": revision,
                "base_public_version": decision["order"],
                "action_bank_version": 0,
            },
            "public": {
                "phase": "card_select",
                "entities": snapshot.entities,
                "relations": snapshot.relations,
                "memory": snapshot.memory,
                "decoder_bank": bank,
                "selection_context": context,
            },
            "legal": {"candidates": candidates},
            "events": [],
        }
        steps.append(
            {
                "frame": clean_frame(frame),
                "candidate_ref": bank[raw_bank.index(label)]["candidate_ref"],
                "forced": len(candidates) == 1,
            }
        )
    return {
        "phase": "card_select",
        "steps": steps,
        "label_path": "reconstructed_from_submitted_result",
    }


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
            return parent is not None and public_teacher(
                parent, seen | {decision["action_id"]}
            )
        return False

    require(
        all(public_teacher(d, set()) for d in decisions),
        "Teacher visibility is unverified (solver/unknown actor); public-input cleaning cannot verify the teacher",
    )
