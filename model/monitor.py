"""Read-only Bootstrap/PPO dashboard; deliberately independent of Torch."""

import argparse
from collections import deque
from datetime import UTC, datetime
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit


WEB = Path(__file__).with_name("monitor_web")
KINDS = {"bootstrap", "ppo"}


def finite_json(value):
    """Old logs can contain NaN; browsers need strict JSON and explicit gaps."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validation_average(validation, a0=False):
    groups = [
        value for key, value in validation.items()
        if key.startswith("A0/") == a0 and isinstance(value, dict)
        and number(value.get("nll")) and number(value.get("branches"))
        and value["branches"] > 0
    ]
    branches = sum(value["branches"] for value in groups)
    return sum(value["nll"] for value in groups) / branches if branches else None


def normalize_record(raw):
    """A point at the recorded optimizer update; old summaries stay sparse."""
    if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str) or raw["kind"] not in KINDS:
        raise ValueError("Not a Bootstrap/PPO training record")
    raw = finite_json(raw)
    epochs = raw.get("metrics", {})
    if isinstance(epochs, dict):
        epochs = [epochs]
    if not isinstance(epochs, list) or any(not isinstance(item, dict) for item in epochs):
        raise ValueError("Metrics must be an object or a list of objects")
    validation = raw.get("validation") or {}
    evaluation = raw.get("evaluation") or {}
    config = raw.get("config") or {}
    if not all(isinstance(value, dict) for value in (validation, evaluation, config)):
        raise ValueError("Invalid training metadata")
    metrics = dict(epochs[-1]) if epochs else {}
    metrics["validation_nll"] = validation_average(validation)
    metrics["a0_validation_nll"] = validation_average(validation, a0=True)
    for name in ("equal_character_win_rate", "worst_character_win_rate",
                 "equal_character_completion_rate"):
        metrics[name] = evaluation.get(name)
    metrics["duration_seconds"] = raw.get("duration_seconds", metrics.get("duration_seconds"))
    update = metrics.get("updates")
    update = update if number(update) and update >= 0 and int(update) == update else None
    return dict(
        kind=raw["kind"], session=raw.get("session"), round=raw.get("round"),
        update=update, record_type="update" if raw.get("record_type") == "update" else "summary",
        time=raw.get("time"), stage=raw.get("stage", raw["kind"]),
        policy_version=raw.get("policy_version"), metrics=metrics,
        optimization_epochs=len(epochs),
        early_stop=any(bool(item.get("early_stop")) for item in epochs),
        validation=validation, evaluation=evaluation, config=config, raw=raw,
    )


class HistoryReader:
    """Incrementally consume complete JSONL lines; never repair the writer's file."""

    def __init__(self, limit=None):
        self.records = deque(maxlen=limit)
        self.offset = 0
        self.signature = None
        self.total = 0
        self.invalid = 0
        self.pending = False
        self.anchor = b""

    def update(self, path):
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if signature == self.signature:
            return
        # Replacement, truncation, or an in-place rewrite of the same length.
        if self.signature and (
            signature[:2] != self.signature[:2] or stat.st_size < self.signature[3]
            or stat.st_size == self.signature[3]
        ):
            self.records.clear()
            self.offset = self.total = self.invalid = 0
        self.pending = False
        with path.open("rb") as stream:
            # A truncate-and-regrow can already be larger than the previous stat.
            # Check the consumed boundary before trusting the stored offset.
            stream.seek(max(0, self.offset - len(self.anchor)))
            if self.offset and stream.read(len(self.anchor)) != self.anchor:
                self.records.clear()
                self.offset = self.total = self.invalid = 0
            stream.seek(self.offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    self.pending = True
                    break
                self.offset = stream.tell()
                if not line.strip():
                    continue
                try:
                    record = normalize_record(json.loads(line))
                except (ValueError, UnicodeDecodeError, RecursionError):
                    self.invalid += 1
                    continue
                self.records.append(record)
                self.total += 1
            stream.seek(max(0, self.offset - 64))
            self.anchor = stream.read(min(self.offset, 64))
        self.signature = signature


def series_record(record):
    """Keep every curve point; omit repeated config and raw JSON from the wire."""
    return {
        **{key: value for key, value in record.items()
           if key not in {"raw", "config", "metrics", "validation", "evaluation"}},
        "metrics": {key: value for key, value in record["metrics"].items()
                    if not isinstance(value, (dict, list))},
        "config": {"target_kl": record["config"].get("target_kl")},
    }


class MonitorStore:
    def __init__(self, root, limit=None):
        self.root = Path(root).resolve()
        self.limit = limit
        self.lock = threading.Lock()
        self.readers = {}
        self.update_readers = {}
        self.directories = {}
        self.discovery_time = None
        self.discovery_errors = 0

    def safe_file(self, directory, name):
        path = directory / name
        # Do not follow a log symlink into a checkpoint or outside the selected root.
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Log path is outside the monitored directory")
        return path

    def discover(self):
        now = time.monotonic()
        if self.discovery_time is not None and now - self.discovery_time < 5:
            return
        directories = {}
        self.discovery_errors = 0
        pending = [self.root]
        while pending:
            directory = pending.pop()
            try:
                # DirEntry avoids a stat per trajectory in large data directories.
                with os.scandir(directory) as entries:
                    children = list(entries)
                names = {entry.name for entry in children if not entry.is_symlink()}
                if {"history.jsonl", "updates.jsonl", "status.json"} & names:
                    key = directory.relative_to(self.root).as_posix()
                    directories[key] = directory
                pending.extend(
                    Path(entry.path) for entry in children
                    if not entry.name.startswith(".") and entry.name not in {
                        "node_modules", "__pycache__", "current"
                    } and entry.is_dir(follow_symlinks=False)
                )
            except OSError:
                self.discovery_errors += 1
        self.directories = directories
        self.readers = {key: reader for key, reader in self.readers.items() if key in directories}
        self.update_readers = {key: reader for key, reader in self.update_readers.items() if key in directories}
        self.discovery_time = now

    def read_run(self, key):
        directory = self.directories[key]
        warnings = []
        status = {}
        try:
            path = self.safe_file(directory, "status.json")
            if path.is_file():
                status = finite_json(json.loads(path.read_text(encoding="utf-8")))
                if not isinstance(status, dict):
                    raise ValueError("Status must be an object")
        except (OSError, ValueError, RecursionError):
            status = {}
            warnings.append("状态文件暂时无法读取。")
        for readers, name in ((self.readers, "history.jsonl"), (self.update_readers, "updates.jsonl")):
            reader = readers.setdefault(key, HistoryReader(self.limit))
            try:
                path = self.safe_file(directory, name)
                if path.is_file():
                    reader.update(path)
                else:
                    reader = readers[key] = HistoryReader(self.limit)
            except (OSError, ValueError):
                warnings.append(f"{name} 暂时无法读取，保留上次读取的记录。")
            if reader.invalid:
                warnings.append(f"{name} 跳过 {reader.invalid} 条损坏或不支持的记录。")
            if reader.pending:
                warnings.append(f"{name} 最后一条记录尚未写完，等待下一次刷新。")
        summaries = self.readers[key]
        updates = self.update_readers[key]
        reader = updates if updates.records else summaries
        latest = reader.records[-1] if reader.records else None
        summary = summaries.records[-1] if summaries.records else None
        kind = status.get("kind") or (latest or {}).get("kind")
        if not isinstance(kind, str) or kind not in KINDS:
            return None
        missing_updates = sum(record["update"] is None for record in reader.records)
        if missing_updates:
            warnings.append(f"{missing_updates} 条记录缺少更新次数，不在更新曲线上绘制。")
        return dict(
            id=key, name=directory.name, path=str(directory), kind=kind,
            status=status, updated_at=status.get("time") or (latest or {}).get("time"),
            total_records=reader.total, retained_records=len(reader.records),
            latest=latest, summary=summary, source="updates" if reader is updates else "history",
            warnings=warnings,
        )

    def list_runs(self):
        with self.lock:
            self.discover()
            runs = [run for key in self.directories if (run := self.read_run(key))]
            runs.sort(key=lambda run: str(run["updated_at"] or ""), reverse=True)
            return dict(
                root=str(self.root), runs=runs, limit=self.limit,
                discovery_errors=self.discovery_errors,
                server_time=datetime.now(UTC).isoformat(),
            )

    def detail(self, key, *, compact=False):
        with self.lock:
            self.discover()
            if key not in self.directories:
                raise KeyError(key)
            run = self.read_run(key)
            if run is None:
                raise KeyError(key)
            reader = self.update_readers[key] if run["source"] == "updates" else self.readers[key]
            records = list(reader.records)
            detail = dict(run, records=records, summaries=list(self.readers[key].records))
            if compact:
                detail["records"] = [series_record(record) for record in records]
                # The table already shows only 200 rows; curves and CSV use all
                # scalar metrics, including the first ever recorded update.
                detail["recent_records"] = records[-200:]
            return detail


def create_server(root, host="127.0.0.1", port=8765, limit=None):
    store = MonitorStore(root, limit)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, body, content_type):
            compressed = len(body) > 4096 and any(
                coding.strip() == "gzip" for coding in self.headers.get("Accept-Encoding", "").split(",")
            )
            if compressed:
                body = gzip.compress(body, compresslevel=1, mtime=0)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Vary", "Accept-Encoding")
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; "
                             "style-src 'self'; script-src 'self'; connect-src 'self'; "
                             "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json(self, status, value):
            body = json.dumps(finite_json(value), ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.respond(status, body, "application/json; charset=utf-8")

        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path == "/api/runs":
                    self.json(200, store.list_runs())
                elif url.path == "/api/run":
                    query = parse_qs(url.query)
                    key = query.get("id", [""])[0]
                    self.json(200, store.detail(key, compact=query.get("compact") == ["1"]))
                elif url.path in {"/", "/index.html", "/app.js", "/style.css", "/favicon.svg"}:
                    name = "index.html" if url.path == "/" else url.path[1:]
                    types = {".html": "text/html", ".js": "text/javascript",
                             ".css": "text/css", ".svg": "image/svg+xml"}
                    self.respond(200, (WEB / name).read_bytes(), types[Path(name).suffix] + "; charset=utf-8")
                else:
                    self.json(404, {"error": "未找到页面。"})
            except KeyError:
                self.json(404, {"error": "训练目录不存在或尚未写入有效记录。"})
            except (OSError, ValueError):
                self.json(503, {"error": "暂时无法读取训练数据，请稍后刷新。"})

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only Bootstrap/PPO training dashboard")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="Training root or a single output directory")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit per run (default: 0, all history)")
    args = parser.parse_args(argv)
    if not args.runs.is_dir():
        parser.error(f"Training directory does not exist: {args.runs}")
    if args.limit < 0:
        parser.error("--limit must be zero (all history) or positive")
    try:
        server = create_server(args.runs, args.host, args.port, args.limit or None)
    except OSError as error:
        parser.error(str(error))
    print(f"Spire training monitor: http://{args.host}:{server.server_port}", flush=True)
    print(f"Reading {args.runs.resolve()} (read only). Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
