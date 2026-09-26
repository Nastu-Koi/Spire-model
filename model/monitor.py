"""Read-only dashboard; training and the web server run independently."""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def snapshot(root):
    result = []
    for path in sorted(Path(root).rglob("status.json")):
        try:
            status = json.loads(path.read_text())
            if "session" not in status or "kind" not in status:
                continue
            if status["kind"] == "bc":
                status["kind"] = "bootstrap"
            rows = []
            history = path.with_name("history.jsonl")
            if history.exists():
                for line in history.read_text().splitlines():
                    try:
                        row = json.loads(line)
                        if row.get("kind") == "bc":
                            row["kind"] = "bootstrap"
                        rows.append(row)
                    except json.JSONDecodeError:
                        pass  # A writer may be between write() and flush().
            if status["state"] in {"starting", "running", "collecting", "updating"}:
                try:
                    os.kill(status["pid"], 0)
                except ProcessLookupError:
                    status["state"] = "interrupted"
                except PermissionError:
                    pass
            result.append(
                {
                    "name": str(path.parent.relative_to(root)),
                    "status": status,
                    "history": rows,
                }
            )
        except (OSError, ValueError, KeyError):
            continue
    return result


def serve(root, host="127.0.0.1", port=8765):
    root = Path(root).resolve()
    html = (Path(__file__).parent / "web/index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/api/runs":
                body = json.dumps(
                    snapshot(root), ensure_ascii=False, allow_nan=False
                ).encode()
                content_type = "application/json; charset=utf-8"
            elif route == "/":
                body, content_type = html, "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer((host, port), Handler) as server:
        print(
            f"Training monitor: http://{host}:{server.server_port}  root={root}",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
