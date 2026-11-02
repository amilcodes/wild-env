"""Stateless structured decisions through the existing authenticated RSG CLI bridges.

The bridge isolates the CLI on the logged-in host. Only a public observation,
prompt and dispatch schema cross it; no simulation files or evaluator state do.
"""

from __future__ import annotations

import hashlib
import os
import re
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from aeolus.agents.adapters import MAX_RESPONSE_BYTES, AgentFailure, AgentReply, AgentSpec
from aeolus.agents.contract import DISPATCH_SCHEMA, SYSTEM_PROMPT, canonical_json, digest, parse_json

CLI_INSTRUCTION = """\nTransport: return exactly one JSON object matching the supplied dispatch schema.
That object is the input to submit_dispatch; the harness applies it once and
advances time. Do not invoke shell, file, browser, search, MCP or other tools.
Use only the observation in this request, including its memory. This is a fresh
stateless decision: no conversation history, repository or external information
is available. For schema-based structured output, emit the structured answer
directly; do not first repeat it as prose. Return a short operational rationale,
not a reasoning transcript. Never invent a task ID or omit a resource.
"""


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _failure(text: str) -> str:
    text = text.lower()
    if any(x in text for x in ("usage limit", "rate_limit", "rate limit", "quota")):
        return "cli_usage_limit"
    if any(x in text for x in ("authenticate", "authentication", "oauth session expired")):
        return "cli_authentication_failed"
    if any(x in text for x in ("not supported", "model_not_found", "does not exist")):
        return "cli_model_unavailable"
    return "cli_request_failed"


def decode_cli(provider: str, stdout: str, stderr: str, returncode: int) -> AgentReply:
    """Parse final structured output and usage; never repair or substitute actions."""
    reply = AgentReply(None, {"input_tokens": None, "output_tokens": None})
    reply.metadata = {
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
        "returncode": returncode,
        "tools_used": [],
        "reported_models": [],
    }
    try:
        events = [parse_json(line) for line in stdout.splitlines() if line.strip()]
        if any(not isinstance(event, dict) for event in events):
            raise ValueError("event must be an object")
        models: set[str] = set()
        tools: set[str] = set()
        if provider == "codex_cli":
            answers, completed = [], []
            for event in events:
                kind = event.get("type")
                if kind == "thread.started":
                    reply.response_id = event.get("thread_id")
                if kind == "turn.completed":
                    completed.append(event)
                if kind in {"turn.failed", "error"}:
                    reply.error = _failure(canonical_json(event))
                item = event.get("item") or {}
                item_type = item.get("type")
                if kind in {"item.started", "item.updated", "item.completed"} and item_type:
                    if item_type not in {"agent_message", "reasoning"}:
                        tools.add(str(item_type))
                    if kind == "item.completed" and item_type == "agent_message":
                        answers.append(item.get("text"))
                if isinstance(event.get("model"), str):
                    models.add(event["model"])
                if kind == "bridge.provenance":
                    reply.metadata["model_identity_source"] = event.get("source")
                    reply.metadata["reasoning_effort"] = event.get("reasoning_effort")
                    reply.metadata["context_sha256"] = event.get("context_sha256")
            # codex exec also reports its selected model in its stderr header.
            models.update(re.findall(r"(?m)^model:\s*([a-z0-9.\-]+)\s*$", stderr))
            if completed:
                usage = completed[-1].get("usage") or {}
                reply.usage = {key: _count(usage.get(key)) for key in reply.usage}
                reply.metadata["cached_input_tokens"] = _count(usage.get("cached_input_tokens"))
            if len(answers) != 1 or len(completed) != 1:
                reply.error = reply.error or "cli_incomplete_or_multiple_answers"
            else:
                reply.dispatch = parse_json(answers[0])
            reply.metadata["model_rounds"] = len(completed)
        elif provider == "claude_cli":
            results = []
            rounds = 0
            for event in events:
                if event.get("type") == "system" and event.get("subtype") == "init":
                    if isinstance(event.get("model"), str):
                        models.add(event["model"])
                    reply.metadata["cli_version"] = event.get("claude_code_version")
                if event.get("type") == "assistant":
                    rounds += 1
                    message = event.get("message") or {}
                    if isinstance(message.get("model"), str) and message["model"] != "<synthetic>":
                        models.add(message["model"])
                    for block in message.get("content") or []:
                        if block.get("type") == "tool_use" and block.get("name") != "StructuredOutput":
                            tools.add(str(block.get("name")))
                if event.get("type") == "result":
                    results.append(event)
            if len(results) != 1:
                reply.error = "cli_missing_result"
            else:
                result = results[0]
                reply.response_id = result.get("session_id")
                usage = result.get("usage") or {}
                input_parts = [
                    _count(usage.get(key, 0))
                    for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
                ]
                reply.usage = {
                    "input_tokens": sum(input_parts)
                    if all(x is not None for x in input_parts) and "input_tokens" in usage
                    else None,
                    "output_tokens": _count(usage.get("output_tokens")),
                }
                reply.metadata["cached_input_tokens"] = _count(usage.get("cache_read_input_tokens"))
                if result.get("is_error") or result.get("subtype") != "success":
                    reply.error = _failure(canonical_json(result))
                reply.dispatch = result.get("structured_output")
                if reply.dispatch is None and result.get("result") and not reply.error:
                    reply.dispatch = parse_json(result["result"])
                model_usage = result.get("modelUsage") or {}
                models.update(key for key in model_usage if isinstance(key, str))
            reply.metadata["model_rounds"] = rounds
        else:
            raise ValueError("unsupported CLI")
        reply.metadata["reported_models"] = sorted(models)
        reply.metadata["tools_used"] = sorted(tools)
        reply.model = next(iter(models)) if len(models) == 1 else None
        if tools:
            reply.error = "cli_disallowed_tool_use"
        if returncode:
            reply.error = reply.error or _failure(stderr + stdout)
        if reply.dispatch is None:
            reply.error = reply.error or "cli_missing_dispatch"
    except (TypeError, ValueError, KeyError, AttributeError):
        reply.error = "cli_invalid_event_stream"
    return reply


class CliAdapter:
    def __init__(self, spec: AgentSpec):
        self.spec = spec
        self.command = list(spec.command) or ["codex" if spec.provider == "codex_cli" else "claude"]
        self.system = SYSTEM_PROMPT + CLI_INSTRUCTION
        out, _, code = self._execute(self.command + ["--version"], b"", timeout=30)
        if code:
            raise ValueError(f"{spec.name}: CLI/bridge version probe failed")
        self.identity = {
            "cli_version": out.decode().strip()[:120],
            "transport_prompt": CLI_INSTRUCTION,
            "effective_system_sha256": digest(self.system),
            "session_policy": "fresh isolated session for every decision; no resume",
            "output_token_cap": None,
            "model_identity": "reported events/header only; absent metadata stays unknown",
        }

    def _execute(
        self, argv: list[str], body: bytes, *, timeout: float | None = None
    ) -> tuple[bytes, bytes, int]:
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            try:
                process = subprocess.Popen(
                    argv, stdin=subprocess.PIPE, stdout=output, stderr=errors, start_new_session=True
                )
            except OSError as exc:
                raise AgentFailure("cli_launch_failed") from exc
            try:
                process.communicate(body, timeout=timeout or self.spec.timeout_s)
            except subprocess.TimeoutExpired as exc:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise AgentFailure("cli_timeout") from exc
            output.seek(0)
            errors.seek(0)
            out, err = output.read(MAX_RESPONSE_BYTES + 1), errors.read(MAX_RESPONSE_BYTES + 1)
            if max(len(out), len(err)) > MAX_RESPONSE_BYTES:
                raise AgentFailure("cli_response_too_large")
            return out, err, process.returncode

    def act(self, observation: dict[str, Any]) -> AgentReply:
        spec = self.spec
        with tempfile.TemporaryDirectory(prefix="wild-env-decision-") as work:
            if spec.provider == "codex_cli":
                schema = Path(work) / "dispatch.json"
                schema.write_text(canonical_json(DISPATCH_SCHEMA))
                argv = [
                    *self.command,
                    "exec",
                    "--sandbox",
                    "read-only",
                    "-C",
                    work,
                    "--json",
                    "--skip-git-repo-check",
                    "-m",
                    spec.model,
                    "-c",
                    f"model_reasoning_effort={spec.reasoning_effort}",
                    "-c",
                    "service_tier=default",
                    "--output-schema",
                    str(schema),
                ]
                for feature in ("browser_use", "computer_use", "apps"):
                    argv += ["--disable", feature]
                argv += ["-"]
                body = (self.system + "\n\nOBSERVATION\n" + canonical_json(observation)).encode()
            else:
                argv = [
                    *self.command,
                    "--output-format",
                    "stream-json",
                    "--input-format",
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
                    "--thinking",
                    "adaptive",
                    "--effort",
                    spec.reasoning_effort,
                    "--max-turns",
                    "3",
                    "--json-schema",
                    canonical_json(DISPATCH_SCHEMA),
                    "--system-prompt",
                    self.system,
                ]
                body = (
                    canonical_json(
                        {"type": "user", "message": {"role": "user", "content": canonical_json(observation)}}
                    )
                    + "\n"
                ).encode()
            out, err, code = self._execute(argv, body)
        reply = decode_cli(
            spec.provider, out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), code
        )
        reply.metadata["effective_system_sha256"] = self.identity["effective_system_sha256"]
        reply.metadata["cli_version"] = reply.metadata.get("cli_version") or self.identity["cli_version"]
        reported = reply.metadata.get("reported_models", [])
        if any(model != spec.model and not model.startswith(spec.model + "-20") for model in reported):
            reply.error = "cli_model_mismatch"
        return reply
