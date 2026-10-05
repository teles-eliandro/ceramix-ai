#!/usr/bin/env python3
"""Evaluate JS in a target on an arbitrary CDP port (via websocket)."""
import json
import sys

from websocket import create_connection  # websocket-client

PORT = sys.argv[1] if len(sys.argv) > 1 else "9333"
TID = sys.argv[2]
EXPR = sys.argv[3]
TIMEOUT = float(sys.argv[4]) if len(sys.argv) > 4 else 90.0

ws_url = None
import urllib.request
with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list", timeout=10) as r:
    for t in json.load(r):
        if t.get("id") == TID:
            ws_url = t.get("webSocketDebuggerUrl")
if not ws_url:
    print(json.dumps({"error": "target not found"}))
    raise SystemExit(1)

ws = create_connection(ws_url, timeout=TIMEOUT)
ws.send(json.dumps({
    "id": 1, "method": "Runtime.evaluate",
    "params": {"expression": EXPR, "returnByValue": True, "awaitPromise": True},
}))
while True:
    msg = json.loads(ws.recv())
    if msg.get("id") == 1:
        print(json.dumps(msg.get("result", {}).get("result", {}).get("value")))
        break
ws.close()
