"""Monitoring contracts exercised without loading Torch or training a model."""

import gzip
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from model.history import TrainingHistory
from model.monitor import HistoryReader, MonitorStore, create_server, normalize_record


def record(round_number=1, kind="bootstrap", **details):
    return dict(kind=kind, round=round_number, session="test-session",
                metrics={"loss": 0.5, "updates": round_number}, **details)


def write_records(path, records):
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


def updates(detail):
    return detail["series"]["meta"]["update"]


class MetricsTests(unittest.TestCase):
    def test_update_coordinate_is_never_inferred_from_epoch(self):
        raw = record(3)
        raw["metrics"]["updates"] = 10624
        self.assertEqual(normalize_record(raw)["update"], 10624)
        for value in (None, "10624", -1, 1.5, True):
            raw["metrics"]["updates"] = value
            self.assertIsNone(normalize_record(raw)["update"])

    def test_update_duration_is_not_replaced_by_a_round_duration(self):
        raw = record(record_type="update")
        raw["metrics"]["duration_seconds"] = 0.75
        row = normalize_record(raw)
        self.assertEqual(row["record_type"], "update")
        self.assertEqual(row["metrics"]["duration_seconds"], 0.75)

    def test_bootstrap_accuracy_preserves_zero_and_does_not_infer_missing_values(self):
        raw = record(record_type="update")
        self.assertNotIn("accuracy", normalize_record(raw)["metrics"])
        raw["metrics"].update(accuracy=0, accuracy_correct=0, accuracy_decisions=12)
        measured = normalize_record(raw)["metrics"]
        self.assertEqual(measured["accuracy"], 0)
        self.assertEqual(measured["accuracy_decisions"], 12)

    def test_bootstrap_weights_branches_and_does_not_double_count_a0(self):
        row = normalize_record(record(validation={
            "Ironclad/combat_play": {"nll": 6, "branches": 3, "macros": 2},
            "Silent/card_select": {"nll": 30, "branches": 1, "macros": 1},
            "A0/Ironclad/combat_play": {"nll": 4, "branches": 2, "macros": 1},
            "Defect/map": {"nll": 0, "branches": 0},
        }))
        self.assertEqual(row["metrics"]["validation_nll"], 9)
        self.assertEqual(row["metrics"]["a0_validation_nll"], 2)
        self.assertEqual(row["metrics"]["loss"], 0.5)

    def test_a_summary_of_several_epochs_keeps_the_last_and_the_raw_record(self):
        raw = record()
        raw["metrics"] = [{"loss": 4, "updates": 10}, {"loss": 2, "updates": 20}]
        row = normalize_record(raw)
        self.assertEqual(row["metrics"]["loss"], 2)
        self.assertEqual(row["metrics"]["updates"], 20)
        self.assertEqual(row["optimization_epochs"], 2)
        self.assertEqual(row["raw"]["metrics"], raw["metrics"])

    def test_missing_values_are_not_zero_and_the_stage_is_preserved(self):
        row = normalize_record(record(stage="bootstrap"))
        self.assertEqual(row["stage"], "bootstrap")
        self.assertIsNone(row["metrics"]["validation_nll"])
        self.assertNotIn("equal_character_win_rate", row["metrics"])
        self.assertNotIn("evaluation", row)

    def test_records_of_the_removed_ppo_are_not_training_records(self):
        with self.assertRaises(ValueError):
            normalize_record(record(kind="ppo", evaluation={"equal_character_win_rate": 0.25}))

    def test_nonfinite_values_become_json_null(self):
        raw = record()
        raw["metrics"]["loss"] = float("nan")
        raw["metrics"]["entropy"] = float("inf")
        row = normalize_record(raw)
        self.assertIsNone(row["metrics"]["loss"])
        self.assertIsNone(row["raw"]["metrics"]["entropy"])
        json.dumps(row, allow_nan=False)

    def test_malformed_metric_shapes_are_rejected(self):
        for raw in ([], {"kind": []}, {"kind": "other"}, {"kind": "bootstrap", "metrics": [4]},
                    {"kind": "bootstrap", "config": "bad"}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize_record(raw)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "history.jsonl"

    def test_append_partial_utf8_tail_and_resume_without_duplicates(self):
        write_records(self.path, [record(1)])
        reader = HistoryReader()
        reader.update(self.path)
        tail = json.dumps(record(2, note="训练"), ensure_ascii=False).encode("utf-8")
        split = tail.index("训".encode("utf-8")) + 1
        with self.path.open("ab") as stream:
            stream.write(tail[:split])
        reader.update(self.path)
        self.assertTrue(reader.pending)
        self.assertEqual(reader.total, 1)
        self.assertEqual(reader.invalid, 0)
        with self.path.open("ab") as stream:
            stream.write(tail[split:] + b"\n")
        reader.update(self.path)
        reader.update(self.path)
        self.assertFalse(reader.pending)
        self.assertEqual([row["round"] for row in reader.records], [1, 2])

    def test_updates_are_available_before_a_summary_and_not_overwritten_by_it(self):
        writer = TrainingHistory(self.root, "bootstrap", {"head_lr": 3e-5})
        self.addCleanup(writer.close)
        writer.append_update({"updates": 7, "loss": 1.5, "duration_seconds": 0.5},
                             round=2, optimization_epoch=1, stage="bootstrap")
        store = MonitorStore(self.root)
        self.assertEqual(store.list_runs()["runs"][0]["kind"], "bootstrap")
        initial = store.detail(".")
        self.assertEqual(initial["total_records"], 1)
        self.assertEqual(initial["source"], "updates")
        self.assertIsNone(initial["summary"])
        self.assertEqual(initial["latest"]["update"], 7)
        self.assertFalse((self.root / "history.jsonl").exists())
        writer.append_update({"updates": 8, "loss": 0.75}, round=2, stage="bootstrap")
        writer.append(2, [{"updates": 8, "loss": 1.1, "accuracy": 0.5}])
        writer.status("completed", round=2)
        result = store.detail(".")
        self.assertEqual(result["status"]["updates"], 8)
        self.assertEqual(updates(result), [7, 8])
        self.assertEqual(result["series"]["metrics"]["loss"], [1.5, 0.75])
        self.assertEqual(sorted(result["series"]["meta"]), ["policy_version", "round", "session", "stage", "time", "update"])
        self.assertEqual(result["latest"]["metrics"]["loss"], 0.75)
        self.assertEqual(result["summary"]["metrics"]["loss"], 1.1)
        self.assertEqual(result["summary"]["metrics"]["accuracy"], 0.5)

    def test_resume_repairs_only_torn_update_tail_and_preserves_sessions(self):
        writer = TrainingHistory(self.root, "bootstrap", {})
        writer.append_update({"updates": 9, "loss": 0.4}, round=1)
        writer.close()
        with (self.root / "updates.jsonl").open("ab") as stream:
            stream.write(b'{"kind":"bootstrap"')
        resumed = TrainingHistory(self.root, "bootstrap", {})
        self.addCleanup(resumed.close)
        resumed.append_update({"updates": 8, "loss": 0.5}, round=1)
        store = MonitorStore(self.root)
        store.list_runs()
        detail = store.detail(".")
        self.assertEqual(updates(detail), [9, 8])
        # A resumed session whose count went back: both points stay, told apart by session.
        self.assertEqual(detail["series"]["meta"]["session"], [0, 1])
        self.assertEqual(detail["series"]["labels"]["session"], [writer.session, resumed.session])

    def test_update_only_directory_is_discovered(self):
        write_records(self.root / "updates.jsonl", [record(record_type="update")])
        store = MonitorStore(self.root)
        result = store.list_runs()["runs"]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["kind"], "bootstrap")
        self.assertEqual(store.detail(".")["source"], "updates")

    def test_invalid_lines_do_not_hide_later_records(self):
        self.path.write_bytes(b"broken\n\xff\n[]\n{\"kind\": []}\n\n" + json.dumps(record()).encode() + b"\n")
        reader = HistoryReader()
        reader.update(self.path)
        self.assertEqual(reader.invalid, 4)
        self.assertEqual(reader.total, 1)

    def test_nan_written_by_old_trainers_is_a_gap_on_the_curve(self):
        self.path.write_bytes(b'{"kind": "bootstrap", "metrics": {"loss": NaN, "accuracy": Infinity, "entropy": 0.5, "updates": 4}}\n')
        (self.root / "status.json").write_text('{"kind": "bootstrap", "state": "completed"}')
        detail = MonitorStore(self.root).detail(".")
        self.assertEqual(detail["series"]["metrics"], {"entropy": [0.5], "updates": [4.0]})
        self.assertIsNone(detail["latest"]["raw"]["metrics"]["loss"])
        json.dumps(detail, allow_nan=False)

    def test_truncation_regrowth_and_rotation_reset_history(self):
        write_records(self.path, [record(1), record(2)])
        reader = HistoryReader()
        reader.update(self.path)
        write_records(self.path, [record(7)])
        reader.update(self.path)
        self.assertEqual([row["round"] for row in reader.records], [7])
        write_records(self.path, [record(8), record(9), record(10)])
        reader.update(self.path)
        self.assertEqual([row["round"] for row in reader.records], [8, 9, 10])
        replacement = self.root / "replacement.jsonl"
        write_records(replacement, [record(30)])
        replacement.replace(self.path)
        reader.update(self.path)
        self.assertEqual([row["round"] for row in reader.records], [30])
        self.assertEqual(list(reader.series.meta["round"]), [30])

    def test_bounded_retention_still_counts_all_valid_records(self):
        write_records(self.path, [record(i) for i in range(10)])
        reader = HistoryReader(limit=3)
        reader.update(self.path)
        self.assertEqual(reader.total, 10)
        self.assertEqual([row["round"] for row in reader.records], [7, 8, 9])
        self.assertEqual(list(reader.series.meta["update"]), [7, 8, 9])
        self.assertEqual(reader.series.since(0)["start"], 7)

    def test_default_keeps_full_history_as_columns_and_sends_only_what_is_new(self):
        for source in ("history", "updates"):
            with self.subTest(source=source):
                directory = self.root / source
                directory.mkdir()
                path = directory / f"{source}.jsonl"
                first = record(1, config={"head_lr": 3e-5, "logical_batch_size": 32})
                first["metrics"].update(accuracy=0, accuracy_correct=0, accuracy_decisions=8,
                                        learning_rates={"head": 0.0001}, parameter_updates={"head": {"samples": 1}})
                write_records(path, [first, *(record(i) for i in range(2, 2502))])
                store = MonitorStore(directory)
                full = store.detail(".")
                series = full["series"]
                self.assertEqual(full["source"], source)
                self.assertEqual(full["total_records"], 2501)
                self.assertEqual(full["retained_records"], 2501)
                self.assertEqual(series["start"], 0)
                self.assertEqual(updates(full), list(range(1, 2502)))
                # A measured zero is a value; a metric the later records lack is a gap.
                self.assertEqual(series["metrics"]["accuracy"][:2], [0, None])
                self.assertEqual(series["metrics"]["accuracy_decisions"][0], 8)
                self.assertEqual(series["metrics"]["learning_rates.head"][:2], [0.0001, None])
                self.assertFalse(any(name.startswith("parameter_updates") for name in series["metrics"]))
                self.assertEqual(len({len(column) for column in (*series["meta"].values(), *series["metrics"].values())}), 1)
                json.dumps(full, allow_nan=False)
                # Raw records stay for the table's newest rows; a summary log keeps its rounds.
                self.assertEqual(len(full["recent"]), 200)
                self.assertEqual(full["recent"][-1]["raw"], record(2501))
                self.assertEqual(full["latest"], full["recent"][-1])
                if source == "updates":
                    self.assertEqual(len(store.update_readers["."].records), 200)
                else:
                    self.assertEqual(len(full["summaries"]["records"]), 2501)
                    self.assertNotIn("raw", full["summaries"]["records"][0])
                with path.open("a") as stream:
                    stream.write(json.dumps(record(2502)) + "\n")
                held = dict(after=2501, summaries=full["summaries"]["total"], token=full["token"])
                appended = store.detail(".", **held)
                self.assertEqual(appended["token"], full["token"])
                self.assertEqual(appended["series"]["start"], 2501)
                self.assertEqual(updates(appended), [2502])
                self.assertEqual([row["update"] for row in appended["recent"]], [2502])
                self.assertEqual(len(appended["summaries"]["records"]), int(source == "history"))
                self.assertEqual(appended["total_records"], 2502)
                # Positions counted against another log are not trusted.
                self.assertEqual(len(updates(store.detail(".", **dict(held, token="stale")))), 2502)
                self.assertEqual(updates(store.detail(".", after=2502, token=full["token"])), [])

    def test_a_replaced_log_starts_the_client_over(self):
        path = self.root / "updates.jsonl"
        write_records(path, [record(i, record_type="update") for i in range(1, 6)])
        store = MonitorStore(self.root)
        first = store.detail(".")
        write_records(path, [record(i, record_type="update") for i in range(40, 43)])
        again = store.detail(".", after=5, token=first["token"])
        self.assertNotEqual(again["token"], first["token"])
        self.assertEqual(again["series"]["start"], 0)
        self.assertEqual(updates(again), [40, 41, 42])
        path.unlink()
        write_records(self.root / "history.jsonl", [record(50)])
        gone = store.detail(".", after=3, token=again["token"])
        self.assertEqual((gone["source"], updates(gone)), ("history", [50]))
        self.assertNotEqual(gone["token"], again["token"])

    def test_limit_keeps_the_newest_points_and_says_where_they_start(self):
        write_records(self.root / "updates.jsonl", [record(i, record_type="update") for i in range(1, 11)])
        store = MonitorStore(self.root, limit=4)
        detail = store.detail(".")
        self.assertEqual((detail["total_records"], detail["retained_records"]), (10, 4))
        self.assertEqual(detail["series"]["start"], 6)
        self.assertEqual(updates(detail), [7, 8, 9, 10])
        self.assertEqual(updates(store.detail(".", after=2, token=detail["token"])), [7, 8, 9, 10])
        self.assertEqual(updates(store.detail(".", after=9, token=detail["token"])), [10])

    def test_listing_does_not_read_the_logs_of_runs_that_state_their_kind(self):
        for name, kind in (("a", "bootstrap"), ("b", "bootstrap"), ("removed", "ppo")):
            (self.root / name).mkdir()
            write_records(self.root / name / "updates.jsonl", [record(1, kind=kind, record_type="update")])
            (self.root / name / "status.json").write_text(json.dumps({"kind": kind, "state": "updating"}))
        (self.root / "old").mkdir()
        write_records(self.root / "old" / "history.jsonl", [record(3)])
        store = MonitorStore(self.root)
        # A directory of the removed PPO is not a training run.
        self.assertEqual({run["id"]: run["kind"] for run in store.list_runs()["runs"]},
                         {"a": "bootstrap", "b": "bootstrap", "old": "bootstrap"})
        # Only the runs without a status file naming a known kind had to be opened.
        self.assertEqual(sorted(key for key, reader in store.readers.items() if reader.total), ["old"])
        self.assertEqual(store.detail("b")["total_records"], 1)
        with self.assertRaises(KeyError):
            store.detail("removed")

    def test_summaries_keep_the_round_report_without_raw_records(self):
        writer = TrainingHistory(self.root, "bootstrap", {"head_lr": 3e-5})
        self.addCleanup(writer.close)
        writer.append_update({"updates": 3, "loss": 1.0}, round=1, stage="bootstrap")
        writer.append(1, [{"updates": 3, "loss": 1.0, "accuracy": 0.25}], stage="bootstrap", training_runs=7)
        detail = MonitorStore(self.root).detail(".")
        (row,) = detail["summaries"]["records"]
        self.assertEqual((row["update"], row["round"], row["stage"]), (3, 1, "bootstrap"))
        self.assertEqual(row["metrics"]["accuracy"], 0.25)
        self.assertTrue({"raw", "config", "validation"}.isdisjoint(row))
        self.assertEqual(detail["summary"]["raw"]["training_runs"], 7)

    def test_real_writer_status_only_append_and_resumed_session(self):
        writer = TrainingHistory(self.root, "bootstrap", {"head_lr": 0.001})
        self.addCleanup(writer.close)
        store = MonitorStore(self.root)
        self.assertEqual(store.list_runs()["runs"][0]["status"]["state"], "starting")
        writer.append(1, {"loss": 2.0})
        first_session = writer.session
        writer.status("completed", round=1)
        writer = TrainingHistory(self.root, "bootstrap", {"head_lr": 0.0001})
        self.addCleanup(writer.close)
        writer.append(2, {"loss": 1.0})
        writer.status("completed", round=2)
        detail = store.detail(".")
        self.assertEqual(detail["total_records"], 2)
        rounds = detail["summaries"]["records"]
        self.assertEqual(rounds[0]["session"], first_session)
        self.assertNotEqual(rounds[1]["session"], first_session)
        self.assertEqual(detail["latest"]["config"]["head_lr"], 0.0001)

    def test_nested_discovery_and_symlinks_do_not_expose_other_files(self):
        outside = self.root / "outside"
        outside.mkdir()
        write_records(outside / "history.jsonl", [record(99)])
        inside = self.root / "selected"
        inside.mkdir()
        (inside / "linked").symlink_to(outside, target_is_directory=True)
        (inside / "history.jsonl").symlink_to(outside / "history.jsonl")
        (inside / "status.json").write_text('{"kind":"bootstrap","state":"updating"}')
        nested = inside / "experiments" / "run"
        nested.mkdir(parents=True)
        write_records(nested / "history.jsonl", [record(1)])
        store = MonitorStore(inside)
        runs = store.list_runs()["runs"]
        self.assertEqual({run["id"] for run in runs}, {".", "experiments/run"})
        self.assertEqual(store.detail(".")["total_records"], 0)
        self.assertTrue(store.detail(".")["warnings"])
        with self.assertRaises(KeyError):
            store.detail("../outside")

    def test_missing_or_invalid_status_does_not_hide_history(self):
        write_records(self.path, [record()])
        (self.root / "status.json").write_text("{torn", encoding="utf-8")
        store = MonitorStore(self.root)
        run = store.list_runs()["runs"][0]
        self.assertEqual(run["kind"], "bootstrap")
        self.assertEqual(run["status"], {})
        self.assertTrue(run["warnings"])


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        write_records(self.root / "history.jsonl", [record()])
        self.server = create_server(self.root, port=0)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join()
        self.temporary.cleanup()

    def test_assets_and_json_are_served_with_no_store(self):
        for path, content_type in [("/", "text/html"), ("/app.js", "text/javascript"),
                                   ("/style.css", "text/css"), ("/favicon.svg", "image/svg+xml"),
                                   ("/api/runs", "application/json"), ("/api/run?id=.", "application/json")]:
            with self.subTest(path=path), urlopen(self.base + path) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                self.assertIn(content_type, response.headers["Content-Type"])
                self.assertNotIn("Access-Control-Allow-Origin", response.headers)
                self.assertTrue(response.read())

    def test_unknown_paths_and_traversal_cannot_read_files(self):
        for path in ("/../history.jsonl", "/history.jsonl", "/%2e%2e/pyproject.toml",
                     "/api/run?" + urlencode({"id": "../outside"}), "/api/run?id=/etc"):
            with self.subTest(path=path), self.assertRaises(HTTPError) as raised:
                urlopen(self.base + path)
            self.assertEqual(raised.exception.code, 404)

    def test_no_write_endpoint(self):
        before = (self.root / "history.jsonl").read_bytes()
        with self.assertRaises(HTTPError) as raised:
            urlopen(Request(self.base + "/api/run?id=.", data=b"{}", method="POST"))
        self.assertEqual(raised.exception.code, 501)
        self.assertEqual((self.root / "history.jsonl").read_bytes(), before)

    def test_compressed_api_keeps_full_history(self):
        write_records(self.root / "updates.jsonl", [record(i, record_type="update") for i in range(1, 2502)])
        request = Request(self.base + "/api/run?id=.", headers={"Accept-Encoding": "deflate,gzip"})
        with urlopen(request) as response:
            self.assertEqual(response.headers["Content-Encoding"], "gzip")
            compressed = response.read()
        body = gzip.decompress(compressed)
        payload = json.loads(body)
        self.assertLess(len(compressed), len(body) / 5)
        self.assertEqual(len(updates(payload)), 2501)
        self.assertEqual((updates(payload)[0], updates(payload)[-1]), (1, 2501))
        self.assertEqual(len(payload["recent"]), 200)
        # The next poll names what it holds and is sent nothing it already has.
        query = urlencode({"id": ".", "after": 2501, "summaries": payload["summaries"]["total"], "token": payload["token"]})
        with urlopen(self.base + "/api/run?" + query) as response:
            follow = json.loads(response.read())
        self.assertEqual((updates(follow), follow["recent"], follow["summaries"]["records"]), ([], [], []))
        with urlopen(self.base + "/api/run?id=.&after=junk&token=" + payload["token"]) as response:
            self.assertEqual(len(updates(json.loads(response.read()))), 2501)


if __name__ == "__main__":
    unittest.main()
