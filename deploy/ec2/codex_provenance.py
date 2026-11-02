#!/usr/bin/env python3
"""Optional logged-in-host shim for the RSG Codex bridge's --codex argument.

Forward codex exec output unchanged, then attest selected model/effort from only
this newly-created session. Never expose session messages or credential files.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main():
    executable = os.environ.get("WILD_ENV_CODEX_BIN") or shutil.which("codex")
    if not executable:
        raise SystemExit("codex executable not found")
    process = subprocess.run([executable, *sys.argv[1:]], input=sys.stdin.buffer.read(), capture_output=True)
    sys.stdout.buffer.write(process.stdout)
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(process.stderr)
    if sys.argv[1:] == ["--version"] or process.returncode:
        return process.returncode
    thread_id = None
    for line in process.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")
    if not isinstance(thread_id, str) or not re.fullmatch(r"[a-f0-9-]{36}", thread_id):
        return process.returncode
    session_root = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "sessions"
    now = datetime.now(timezone.utc)
    for offset in (0, -1, 1):
        directory = session_root / (now + timedelta(days=offset)).strftime("%Y/%m/%d")
        for path in directory.glob(f"*{thread_id}*.jsonl"):
            for line in path.open():
                entry = json.loads(line)
                if entry.get("type") != "turn_context":
                    continue
                context = entry["payload"]
                public = {
                    "model": context.get("model"),
                    "reasoning_effort": context.get("effort"),
                    "sandbox_policy": context.get("sandbox_policy"),
                }
                checksum = hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()
                print(
                    json.dumps(
                        {
                            "type": "bridge.provenance",
                            "source": "codex_turn_context",
                            "thread_id": thread_id,
                            "context_sha256": checksum,
                            **public,
                        }
                    ),
                    flush=True,
                )
                return process.returncode
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
