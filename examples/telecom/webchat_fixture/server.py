"""Bounded loopback server for the controlled telecom chat fixture."""

from __future__ import annotations

import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/frame.html": "frame.html",
    "/app.js": "app.js",
    "/style.css": "style.css",
}


class FixtureHandler(BaseHTTPRequestHandler):
    """Serve only the fixture's known files; the query never selects a path."""

    def do_GET(self) -> None:
        self._serve(body=True)

    def do_HEAD(self) -> None:
        self._serve(body=False)

    def _serve(self, *, body: bool) -> None:
        path = unquote(urlsplit(self.path).path)
        if path == "/health":
            content, media_type = b"ready", "text/plain; charset=utf-8"
        elif path in FILES:
            file = (ROOT / FILES[path]).resolve()
            if not file.is_relative_to(ROOT):
                self.send_error(404)
                return
            content = file.read_bytes()
            media_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            if media_type.startswith("text/") or media_type == "application/javascript":
                media_type += "; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        """Keep fixture traffic out of test output."""


class FixtureServer:
    """Context-managed local fixture server with a finite readiness and shutdown budget."""

    def __init__(self, timeout_seconds: float = 3.0) -> None:
        if not 0 < timeout_seconds < float("inf"):
            raise ValueError("timeout_seconds must be finite and positive")
        self.timeout_seconds = timeout_seconds
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("fixture server is not running")
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self) -> FixtureServer:
        if self._server is not None:
            raise RuntimeError("fixture server is already running")
        server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        deadline = time.monotonic() + self.timeout_seconds
        try:
            self._thread.start()
            while True:
                try:
                    with urlopen(
                        f"{self.base_url}/health", timeout=min(0.2, self.timeout_seconds)
                    ) as response:
                        if response.read() == b"ready":
                            return self
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("fixture server readiness timed out")
                time.sleep(min(0.01, max(0, deadline - time.monotonic())))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        server, thread = self._server, self._thread
        if server is None:
            return
        self._server = None
        self._thread = None
        if thread is not None and thread.is_alive():
            shutdown = threading.Thread(target=server.shutdown, daemon=True)
            shutdown.start()
            shutdown.join(self.timeout_seconds)
            if shutdown.is_alive():
                server.server_close()
                raise TimeoutError("fixture server shutdown timed out")
        server.server_close()
        if thread is not None:
            thread.join(self.timeout_seconds)
            if thread.is_alive():
                raise TimeoutError("fixture server thread did not exit")

    def __enter__(self) -> FixtureServer:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()


if __name__ == "__main__":
    try:
        with FixtureServer() as fixture:
            print(f"Fixture available at {fixture.base_url}/?variant=success", flush=True)
            threading.Event().wait()
    except KeyboardInterrupt:
        pass
