#!/usr/bin/env python3
"""Serve the ActorII HTML5 surface over HTTP."""

import os
import socket
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from actorii.display.html5 import Button, Checkbox, HTML5Surface


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def run_server(host_ip: str, host_port: int = 8000) -> None:
    surface = HTML5Surface(width=800, height=600, title="ActorII Remote UI")
    surface.register_object(Button(x=60, y=310, width=130, height=34, label="Inspect Object"))
    surface.register_object(Checkbox(x=210, y=315, width=140, height=30, label="Trace Messages"))

    html_content = surface.render()
    index_file = os.path.join(ROOT_DIR, "index.html")
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    class CustomHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=ROOT_DIR, **kwargs)

    server = ReusableThreadingHTTPServer(("0.0.0.0", host_port), CustomHandler)
    print(f"ActorII HTML5 server running at http://{host_ip}:{host_port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    try:
        hostname = socket.gethostname()
        host_ip = socket.gethostbyname(hostname)
    except Exception:
        host_ip = "127.0.0.1"
    run_server(host_ip)
