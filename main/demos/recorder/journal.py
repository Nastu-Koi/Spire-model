"""Import Steam RunRecorder journals without replaying or inventing game actions
(`read_journal`, `convert_journal`, `import_recordings`)."""

import glob
import hashlib
import json
from collections import Counter
from pathlib import Path

from main.data.shards import validate_run
from main.env.control import controller_for
from main.env.public_history import HISTORY_VERSION
from main.env.protocol import CHARACTERS, ProtocolError, SCHEMA
from main.demos.recorder.macros import CANCEL_OPENERS, NONCOMBAT_CANCEL, SELECTION_CANCEL, decision_macro, flatten, verify_teachers
from main.demos.recorder.snapshot import require


RECORDER_SCHEMA = "sts2-run-recorder/v2"


IMPORT_VERSION = "run-recorder-import-v6"


def read_journal(path):
    """Validate the append-only envelope before considering any sample."""
    rows, digest = [], hashlib.sha256()
    with Path(path).open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            require(raw.endswith(b"\n"), f"Truncated journal line {number}")
            row = json.loads(raw)
            require(
                isinstance(row, dict) and row.get("schema") == RECORDER_SCHEMA,
                f"Unsupported recorder schema on line {number}",
            )
            require(row.get("seq") == number, f"Journal sequence gap on line {number}")
            require(
                row.get("run_id") and row.get("segment_id"), "Missing journal identity"
            )
            if rows:
                require(
                    (row["run_id"], row["segment_id"])
                    == (rows[0]["run_id"], rows[0]["segment_id"]),
                    "Mixed journal identities",
                )
            rows.append(row)
    require(rows, "Empty recording")
    return rows, digest.hexdigest()


def convert_journal(path, *, bc_only=False, recorder_version=None):
    rows, digest = read_journal(path)
    require(rows[0]["kind"] == "segment_start", "Missing segment start")
    start = rows[0]["data"]
    ascension = start["state"]["run"].get("ascension")
    require(type(ascension) is int and 0 <= ascension <= 10, "Recording has no ascension from 0 to 10")
    if recorder_version is not None:
        require(
            start.get("recorder_version") == recorder_version,
            "Recorder version does not match filter",
        )
    endings = [row["data"] for row in rows if row["kind"] == "run_ended"]
    if not bc_only:
        require(
            start.get("begins_at_run_start") is True,
            "Resumed fragment; complete-run continuity is unverified",
        )
        require(
            rows[-1]["kind"] == "segment_end",
            "Recording is active or truncated; no segment end",
        )
        require(
            rows[-1]["data"].get("uncommitted_operations") == 0,
            "Recording has uncommitted decisions",
        )
        require(
            rows[-1]["data"].get("state_capture_errors", 0) == 0,
            "Recording contains failed state captures",
        )
        require(
            len(endings) == 1 and type(endings[0].get("victory")) is bool,
            "Run has no unique terminal outcome",
        )
        terminal = endings[0].get("state", {}).get("run", {})
        require(
            terminal.get("ended") is True
            and terminal.get("victory") == endings[0]["victory"]
            and terminal.get("ascension") == ascension,
            "Terminal snapshot does not confirm outcome",
        )
        require(
            rows[-1]["data"].get("run_ended") is True,
            "Segment end does not confirm run completion",
        )
        allowed = {"segment_start", "segment_end", "run_ended", "decision_committed"}
        require(
            all(row["kind"] in allowed for row in rows),
            "Recording contains errors or an unsupported event format",
        )
        require(
            sum(row["kind"] == "segment_start" for row in rows) == 1,
            "Repeated segment start",
        )
    state = start["state"]
    require(len(state["players"]) == 1, "Only single-player recordings are supported")
    character = next(
        (x for x in CHARACTERS if x.upper() == state["players"][0]["character"]), None
    )
    require(character is not None, "Unsupported character")
    decisions = [
        item
        for row in rows
        if row["kind"] == "decision_committed"
        for item in flatten(row["data"])
    ]
    require(decisions, "Recording has no decisions")
    decisions.sort(key=lambda d: d["order"])
    require(
        len({d["action_id"] for d in decisions}) == len(decisions),
        "Repeated decision identity",
    )
    require(
        len({d["order"] for d in decisions}) == len(decisions),
        "Ambiguous decision order",
    )
    actors = Counter(d.get("actor", "unknown") for d in decisions)
    # A solver actor label does not establish what information its search used.
    if not bc_only:
        verify_teachers(decisions)
    contract = {
        "adapter_version": IMPORT_VERSION,
        "observation_schema": "public-state-v7",
        "public_history_version": HISTORY_VERSION,
        "action_schema": "candidate-v0",
        "fixed_ascension": ascension,
        "training_ready": True,
        "game_version": start["game_version"],
        "recorder_version": start["recorder_version"],
        "game_assembly_mvid": next(
            (
                a["mvid"]
                for a in start.get("loaded_assemblies", [])
                if a["name"] == "sts2"
            ),
            None,
        ),
        "effect_coverage": "typed-public-fields-with-explicit-opaque-rules",
        "action_boundary": "recorded-native-ui",
        "selection_cancel": SELECTION_CANCEL,
        "purpose": "offline_behavior_cloning",
    }
    require(contract["game_assembly_mvid"], "Missing recorded game assembly identity")
    macros, automatic, rejected, environment = [], 0, [], []
    previous = None  # (decision, macro appended for it or None)
    for decision in decisions:
        try:
            require(
                decision["state"]["players"][0]["character"] == character.upper(),
                "Character changed within recording",
            )
            require(
                str(decision["state"]["run"].get("seed")) == str(start["seed"]),
                "Seed changed within recording",
            )
            macro = decision_macro(decision, rows[0]["run_id"], contract)
        except ProtocolError as exc:
            if str(exc) != NONCOMBAT_CANCEL:
                if not bc_only:
                    raise
                rejected.append({"action_id": decision.get("action_id"), "reason": str(exc)})
                previous = (decision, None)
                continue
            # The native detour open -> cancel is a no-op: drop both its label and
            # the label of the action that opened it, in either import mode.
            rejected.append({"action_id": decision.get("action_id"), "reason": NONCOMBAT_CANCEL})
            # Only rest-site options, shop purchases and rewards open such a
            # selection; an unrecorded opener leaves nothing else to drop.
            if (previous is not None and previous[1] is not None and macros
                    and macros[-1] is previous[1] and previous[1]["phase"] in CANCEL_OPENERS):
                macros.pop()
                rejected.append({"action_id": previous[0].get("action_id"),
                                 "reason": "Opened a canceled non-combat selection"})
            previous = (decision, None)
            continue
        except (
            ProtocolError,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            AttributeError,
        ) as exc:
            if not bc_only:
                raise
            rejected.append(
                {"action_id": decision.get("action_id"), "reason": str(exc)}
            )
            previous = (decision, None)
            continue
        controllers = [controller_for(s["frame"]["public"]["phase"], s["frame"]["legal"]["candidates"])
                       for s in macro["steps"]]
        if any(controllers):
            require(all(controllers), "Macro mixes policy and environment steps")
            environment.extend(dict(actor=decision.get("actor", "unknown"),
                                    action_id=decision["action_id"], **step) for step in macro["steps"])
            automatic += len(macro["steps"])
            previous = (decision, None)
        elif all(s["forced"] for s in macro["steps"]):
            automatic += len(macro["steps"])
            previous = (decision, None)
        else:
            macros.append(macro)
            previous = (decision, macro)
    require(macros or environment, "Recording contains no branching decisions")
    run = {
        "schema": SCHEMA,
        "source": "demonstration",
        "teacher_visibility": "public",
        "status": "complete",
        "victory": endings[0]["victory"] if not bc_only else None,
        "ascension": ascension,
        "character": character,
        "run_id": rows[0]["run_id"],
        "seed": str(start["seed"]),
        "contract": contract,
        "macros": macros,
        "automatic_steps": automatic,
        "environment_actions": environment,
        "provenance": {
            "importer": IMPORT_VERSION,
            "raw_sha256": digest,
            "segment_id": rows[0]["segment_id"],
            "actors": dict(actors),
        },
    }
    if bc_only:
        # These are supervised decisions, never a claim of complete on-policy play.
        run.update(
            source="recorder_bc",
            teacher_visibility="unverified",
            status="partial",
            victory=None,
            run_id=rows[0]["run_id"] + ":" + rows[0]["segment_id"],
        )
        run["provenance"].update(
            bc_only=True,
            original_run_id=rows[0]["run_id"],
            rejected_decisions=rejected,
            decisions_seen=len(decisions),
        )
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
    with (
        (directory / "accepted.jsonl").open("x", encoding="utf-8") as good,
        (directory / "quarantine.jsonl").open("x", encoding="utf-8") as bad,
    ):
        for path in sorted(files):
            try:
                if recorder_version is not None:
                    with path.open() as stream:
                        header = json.loads(next(stream, ""))
                    if (
                        header.get("data", {}).get("recorder_version")
                        != recorder_version
                    ):
                        counts["version_filtered_files"] += 1
                        continue
                run = convert_journal(
                    path, bc_only=bc_only, recorder_version=recorder_version
                )
                require(
                    run["run_id"] not in identities, "Duplicate complete-run identity"
                )
                identities.add(run["run_id"])
                good.write(json.dumps(run, ensure_ascii=False, allow_nan=False) + "\n")
                counts["accepted_runs"] += 1
                counts["macros"] += len(run["macros"])
                counts["rejected_decisions"] += len(
                    run["provenance"].get("rejected_decisions", [])
                )
                for macro in run["macros"]:
                    counts[run["character"] + "/" + macro["phase"]] += 1
            except (
                ProtocolError,
                ValueError,
                KeyError,
                TypeError,
                OSError,
                IndexError,
                AttributeError,
            ) as exc:
                reason = str(exc)
                reasons[reason] += 1
                counts["quarantined_files"] += 1
                bad.write(
                    json.dumps(
                        {"path": str(path.resolve()), "reason": reason},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    summary = {
        "importer": IMPORT_VERSION,
        "bc_only": bc_only,
        "recorder_version_filter": recorder_version,
        "files": len(files),
        "accepted_runs": 0,
        **dict(counts),
        "quarantine_reasons": dict(reasons),
        "accepted": str(directory / "accepted.jsonl"),
        "quarantine": str(directory / "quarantine.jsonl"),
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
