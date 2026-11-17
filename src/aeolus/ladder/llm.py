"""Model adapters for the plan benchmark.

One fresh, tool-less session per instance. The model sees the system prompt and
the public observation and must return the plan schema. Nothing is repaired:
parse failures and disallowed tool use are recorded as such.

Providers:
- ``claude_cli`` / ``codex_cli``: the locally logged-in Claude Code / Codex CLIs
  (subscription-backed). Reuses the event-stream decoder of the v2 bridge adapter.
- ``command``: any argv that reads ``{"system", "observation", "schema"}`` JSON on
  stdin and prints the plan JSON on stdout (for API-key clients or local models).
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aeolus.agents.adapters import MAX_RESPONSE_BYTES, AgentReply
from aeolus.agents.cli_adapters import decode_cli
from aeolus.agents.contract import canonical_json, digest, parse_json
from aeolus.ladder.plan import PLAN_SCHEMA
from aeolus.ladder.prompt import SYSTEM_PROMPT

TRANSPORT = """
Transport: this is a single stateless request. Do not call shell, file, browser,
search, MCP or any other tools; everything you need is in the observation. Return
exactly one JSON object matching the supplied schema and nothing else.
"""
PROVIDERS = ("claude_cli", "codex_cli", "command")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model: str
    effort: str | None = "high"
    timeout_s: float = 900.0
    command: tuple[str, ...] = ()
    tier: str = "unlabelled"  # e.g. "frontier" or "small"; descriptive only, never used in scoring

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS}")
        if self.provider == "command" and not self.command:
            raise ValueError("command provider needs an argv list")
        if not 0 < self.timeout_s <= 3600:
            raise ValueError("timeout_s must be in (0, 3600]")

    def public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Call:
    reply: AgentReply
    latency_s: float
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _run(argv: list[str], body: bytes, timeout: float, cwd: str | None = None) -> tuple[str, str, int]:
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        process = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=out, stderr=err, start_new_session=True, cwd=cwd
        )
        try:
            process.communicate(body, timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise TimeoutError("model call timed out") from None
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(MAX_RESPONSE_BYTES + 1), err.read(MAX_RESPONSE_BYTES + 1)
        if max(len(stdout), len(stderr)) > MAX_RESPONSE_BYTES:
            raise ValueError("model response too large")
        return stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace"), process.returncode


class PlanModel:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.system = SYSTEM_PROMPT + TRANSPORT
        self.identity = {"system_sha256": digest(self.system), "schema_sha256": digest(PLAN_SCHEMA)}

    def act(self, observation: dict[str, Any]) -> Call:
        spec = self.spec
        began = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="wild-ladder-") as work:
                if spec.provider == "codex_cli":
                    schema = Path(work) / "plan_schema.json"
                    schema.write_text(canonical_json(PLAN_SCHEMA))
                    argv = [
                        "codex",
                        "exec",
                        "--sandbox",
                        "read-only",
                        "-C",
                        work,
                        "--json",
                        "--skip-git-repo-check",
                        "--ephemeral",
                        "-m",
                        spec.model,
                        "--output-schema",
                        str(schema),
                    ]
                    if spec.effort:
                        argv += ["-c", f"model_reasoning_effort={spec.effort}"]
                    for feature in ("browser_use", "computer_use", "apps"):
                        argv += ["--disable", feature]
                    argv += ["-"]
                    body = (self.system + "\n\nOBSERVATION\n" + canonical_json(observation)).encode()
                    stdout, stderr, code = _run(argv, body, spec.timeout_s, cwd=work)
                    reply = decode_cli("codex_cli", stdout, stderr, code)
                elif spec.provider == "claude_cli":
                    argv = [
                        "claude",
                        "-p",
                        "--output-format",
                        "stream-json",
                        "--verbose",
                        "--tools",
                        "",
                        "--permission-mode",
                        "dontAsk",
                        "--setting-sources=",
                        "--no-session-persistence",
                        "--disable-slash-commands",
                        "--strict-mcp-config",
                        "--mcp-config",
                        '{"mcpServers":{}}',
                        "--model",
                        spec.model,
                        "--json-schema",
                        canonical_json(PLAN_SCHEMA),
                        "--system-prompt",
                        self.system,
                    ]
                    if spec.effort:
                        argv += ["--effort", spec.effort]
                    body = ("OBSERVATION\n" + canonical_json(observation)).encode()
                    stdout, stderr, code = _run(argv, body, spec.timeout_s, cwd=work)
                    reply = decode_cli("claude_cli", stdout, stderr, code)
                else:
                    body = canonical_json(
                        {"system": self.system, "observation": observation, "schema": PLAN_SCHEMA}
                    ).encode()
                    stdout, stderr, code = _run(list(spec.command), body, spec.timeout_s, cwd=work)
                    reply = AgentReply(None, {"input_tokens": None, "output_tokens": None}, model=spec.model)
                    if code:
                        reply.error = "command_failed"
                    else:
                        try:
                            reply.dispatch = parse_json(stdout)
                        except ValueError:
                            reply.error = "command_invalid_json"
        except TimeoutError:
            return Call(AgentReply(None), time.monotonic() - began, "timeout")
        except (OSError, ValueError) as exc:
            return Call(AgentReply(None), time.monotonic() - began, f"transport: {type(exc).__name__}")
        error = reply.error
        reported = reply.metadata.get("reported_models", []) if reply.metadata else []
        if (
            spec.provider != "command"
            and reported
            and any(m != spec.model and not m.startswith(spec.model + "-20") for m in reported)
        ):
            # Claude Code may bill a small auxiliary model; only the authoring model must match.
            authoring = [m for m in reported if m == spec.model or m.startswith(spec.model + "-20")]
            if not authoring:
                error = error or "model_mismatch"
        return Call(reply, time.monotonic() - began, error)
