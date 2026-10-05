#!/usr/bin/env python3
"""Inspeciona serviços Render via API (token lido do ambiente)."""
import os
import json
import urllib.request

TOKEN = os.environ.get("RENDER_API_KEY", "")
if not TOKEN:
    raise SystemExit("SEM RENDER_API_KEY no ambiente")


def api(path):
    req = urllib.request.Request(
        "https://api.render.com/v1" + path,
        headers={"Authorization": "Bearer " + TOKEN, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)


def show(sid):
    s = api("/services/" + sid)
    print("=== " + str(s.get("name")) + " (" + sid + ") ===")
    for k in ["name", "type", "repo", "branch", "autoDeploy", "suspended", "createdAt", "updatedAt"]:
        print(f"  {k}: {s.get(k)}")
    d = s.get("serviceDetails", {}) or {}
    for k in ["url", "plan", "region", "runtime", "healthCheckPath"]:
        print(f"  {k}: {d.get(k)}")
    env = d.get("env") or []
    print(f"  env vars: {len(env)}")
    for e in env:
        print("     -", e.get("key") if isinstance(e, dict) else e)
    print("  --- ultimos deploys ---")
    for it in api("/services/" + sid + "/deploys?limit=5"):
        dp = it.get("deploy", it)
        print("   ", dp.get("id"), dp.get("status"), dp.get("createdAt"),
              str(dp.get("commit", {}).get("message", ""))[:70])
    print()


if __name__ == "__main__":
    import sys
    for sid in sys.argv[1:]:
        show(sid)
