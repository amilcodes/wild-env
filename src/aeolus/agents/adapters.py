"""Observation-only baselines and small, bounded model/command adapters."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

from aeolus.agents.contract import (
    DISPATCH_SCHEMA,
    SYSTEM_PROMPT,
    TOOL_DESCRIPTION,
    canonical_json,
    parse_json,
)

MAX_RESPONSE_BYTES = 1_048_576
BASELINES = ("hold", "nearest", "greedy")


@dataclass(frozen=True)
class AgentSpec:
    name: str
    provider: str = "baseline"
    model: str = "hold"
    base_url: str | None = None
    api_key_env: str | None = None
    command: tuple[str, ...] = ()
    max_output_tokens: int = 2048
    timeout_s: float = 60.0
    max_calls: int = 128
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.model.strip():
            raise ValueError("agent name and model must be nonempty")
        if self.provider not in {"baseline", "openai", "anthropic", "compatible", "command"}:
            raise ValueError("unknown agent provider")
        if self.provider == "baseline" and self.model not in BASELINES:
            raise ValueError("unknown desk baseline")
        if self.provider == "command" and (not self.command or isinstance(self.command, str)):
            raise ValueError("command must be a nonempty argv list, never a shell string")
        if not all(isinstance(arg, str) and arg for arg in self.command):
            raise ValueError("command argv must contain nonempty strings")
        if (
            type(self.max_output_tokens) is not int
            or self.max_output_tokens < 1
            or type(self.max_calls) is not int
            or self.max_calls < 1
            or not math.isfinite(self.timeout_s)
            or not 0 < self.timeout_s <= 3600
        ):
            raise ValueError("output tokens, call count, and timeout must be positive (timeout <= 3600)")
        if self.provider == "compatible" and not self.base_url:
            raise ValueError("compatible provider requires an explicit base_url")
        if self.base_url:
            url = urllib.parse.urlsplit(self.base_url)
            local_http = url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}
            if (url.scheme != "https" and not local_http) or not url.hostname:
                raise ValueError("base_url must use HTTPS (HTTP is allowed only on loopback)")
            if url.username or url.password or url.query or url.fragment:
                raise ValueError("base_url cannot contain credentials, query, or fragment")
        if self.reasoning_effort and self.provider != "openai":
            raise ValueError("reasoning_effort is supported by the openai adapter only")

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)  # Environment variable names only; never resolved credentials.


@dataclass
class AgentReply:
    dispatch: Any
    usage: dict[str, int | None] = field(default_factory=dict)
    response_id: str | None = None
    model: str | None = None
    error: str | None = None


class AgentFailure(RuntimeError):
    """A safe, bounded diagnostic suitable for a result artifact."""


def baseline_dispatch(observation: dict[str, Any], policy: str) -> dict[str, Any]:
    """Heuristics consume exactly the same serialized packet as the LLM."""
    if policy not in BASELINES:
        raise ValueError("unknown baseline")
    tasks = {task["id"]: task for task in observation["tasks"]}
    remaining = {task_id: task["capacity"] for task_id, task in tasks.items()}
    assignments = []
    for resource in observation["resources"]:
        legal = [task_id for task_id in resource["legal_tasks"] if task_id and remaining[task_id] > 0]
        selected = 0
        if legal and policy == "nearest":
            selected = min(legal, key=lambda task_id: (resource["travel_min"][str(task_id)], task_id))
        elif legal and policy == "greedy":

            def utility(task_id: int) -> float:
                task = tasks[task_id]
                information = 0.7 * task["uncertainty"] if task["kind"] == "OBSERVE" else 0.0
                return (
                    task["heuristic_value"] * (1.0 - 0.35 * task["uncertainty"])
                    + information
                    - 0.07 * resource["travel_min"][str(task_id)]
                )

            selected = max([0, *legal], key=lambda task_id: (utility(task_id), -task_id))
        if selected:
            remaining[selected] -= 1
        assignments.append({"resource_id": resource["id"], "task_id": selected})
    return {
        "decision_id": observation["decision_id"],
        "assignments": assignments,
        "rationale": f"{policy} observation-only baseline",
        "memory": "",
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class AgentAdapter:
    def __init__(self, spec: AgentSpec):
        self.spec = spec
        self._key: str | None = None
        if spec.provider not in {"baseline", "command"}:
            env = (
                spec.api_key_env
                or {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "compatible": "OPENAI_API_KEY",
                }[spec.provider]
            )
            self._key = os.environ.get(env)
            # Local inference endpoints may intentionally have no authentication.
            local = spec.base_url and urllib.parse.urlsplit(spec.base_url).hostname in {
                "localhost",
                "127.0.0.1",
                "::1",
            }
            if not self._key and not local:
                raise ValueError(f"agent {spec.name}: set environment variable {env}")

    def act(self, observation: dict[str, Any]) -> AgentReply:
        if self.spec.provider == "baseline":
            return AgentReply(
                baseline_dispatch(observation, self.spec.model),
                {"input_tokens": 0, "output_tokens": 0},
                model=self.spec.model,
            )
        if self.spec.provider == "command":
            return self._command(observation)
        return self._remote(observation)

    def _command(self, observation: dict[str, Any]) -> AgentReply:
        request = {"system": SYSTEM_PROMPT, "observation": observation, "dispatch_schema": DISPATCH_SCHEMA}
        # File-backed stdout avoids unbounded RAM capture. Never execute a shell.
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            try:
                process = subprocess.Popen(
                    self.spec.command,
                    stdin=subprocess.PIPE,
                    stdout=output,
                    stderr=errors,
                    start_new_session=True,
                )
            except OSError as exc:
                raise AgentFailure("command_launch_failed") from exc
            try:
                process.communicate((canonical_json(request) + "\n").encode(), timeout=self.spec.timeout_s)
            except subprocess.TimeoutExpired as exc:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise AgentFailure("command_timeout") from exc
            if process.returncode:
                raise AgentFailure(f"command_exit_{process.returncode}")
            output.seek(0)
            raw = output.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise AgentFailure("response_too_large")
        try:
            dispatch = parse_json(raw.decode())
        except (UnicodeDecodeError, ValueError) as exc:
            raise AgentFailure("invalid_json") from exc
        return AgentReply(dispatch, {"input_tokens": None, "output_tokens": None}, model=self.spec.model)

    def _request(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        request = urllib.request.Request(
            url, data=canonical_json(body).encode(), headers={"Content-Type": "application/json", **headers}
        )
        try:
            with urllib.request.build_opener(_NoRedirect()).open(
                request, timeout=self.spec.timeout_s
            ) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            # Provider error bodies/headers can contain secrets; do not retain them.
            raise AgentFailure(f"provider_http_{exc.code}") from None
        except (OSError, urllib.error.URLError) as exc:
            raise AgentFailure("provider_transport_error") from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise AgentFailure("response_too_large")
        try:
            payload = parse_json(raw.decode())
        except (UnicodeDecodeError, ValueError) as exc:
            raise AgentFailure("provider_invalid_json") from exc
        if not isinstance(payload, dict):
            raise AgentFailure("provider_invalid_envelope")
        return payload

    def _remote(self, observation: dict[str, Any]) -> AgentReply:
        spec = self.spec
        content = canonical_json(observation)
        function = {
            "name": "submit_dispatch",
            "description": TOOL_DESCRIPTION,
            "parameters": DISPATCH_SCHEMA,
            "strict": True,
        }
        if spec.provider == "openai":
            base = spec.base_url or "https://api.openai.com/v1"
            body: dict[str, Any] = {
                "model": spec.model,
                "instructions": SYSTEM_PROMPT,
                "input": [{"role": "user", "content": content}],
                "store": False,
                "tools": [{"type": "function", **function}],
                "tool_choice": {"type": "function", "name": "submit_dispatch"},
                "parallel_tool_calls": False,
                "max_output_tokens": spec.max_output_tokens,
            }
            if spec.reasoning_effort:
                body["reasoning"] = {"effort": spec.reasoning_effort}
            payload = self._request(
                base.rstrip("/") + "/responses", body, {"Authorization": f"Bearer {self._key}"}
            )
        elif spec.provider == "anthropic":
            base = spec.base_url or "https://api.anthropic.com/v1"
            body = {
                "model": spec.model,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": spec.max_output_tokens,
                "tools": [
                    {
                        "name": "submit_dispatch",
                        "description": TOOL_DESCRIPTION,
                        "input_schema": DISPATCH_SCHEMA,
                    }
                ],
                "tool_choice": {"type": "tool", "name": "submit_dispatch", "disable_parallel_tool_use": True},
            }
            payload = self._request(
                base.rstrip("/") + "/messages",
                body,
                {"x-api-key": self._key or "", "anthropic-version": "2023-06-01"},
            )
        else:
            body = {
                "model": spec.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "tools": [{"type": "function", "function": function}],
                "tool_choice": {"type": "function", "function": {"name": "submit_dispatch"}},
                "max_tokens": spec.max_output_tokens,
            }
            headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
            payload = self._request((spec.base_url or "").rstrip("/") + "/chat/completions", body, headers)
        return decode_reply(spec.provider, payload)


def decode_reply(provider: str, payload: dict[str, Any]) -> AgentReply:
    """Keep token accounting even when the model refuses or produces an invalid tool call."""
    usage = payload.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    input_key, output_key = (
        ("prompt_tokens", "completion_tokens")
        if provider == "compatible"
        else ("input_tokens", "output_tokens")
    )
    tokens = {"input_tokens": usage.get(input_key), "output_tokens": usage.get(output_key)}
    tokens = {key: value if type(value) is int and value >= 0 else None for key, value in tokens.items()}
    if provider == "anthropic" and tokens["input_tokens"] is not None:
        for key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            cached = usage.get(key, 0)
            if type(cached) is not int or cached < 0:
                tokens["input_tokens"] = None
                break
            tokens["input_tokens"] += cached
    reply = AgentReply(
        None,
        tokens,
        payload.get("id") if isinstance(payload.get("id"), str) else None,
        payload.get("model") if isinstance(payload.get("model"), str) else None,
    )
    try:
        if provider == "openai":
            if payload.get("status") != "completed":
                raise ValueError("incomplete")
            calls = [item for item in payload.get("output", []) if item.get("type") == "function_call"]
            if len(calls) != 1 or calls[0].get("name") != "submit_dispatch":
                raise ValueError("expected one tool call")
            reply.dispatch = parse_json(calls[0]["arguments"])
        elif provider == "anthropic":
            if payload.get("stop_reason") != "tool_use":
                raise ValueError("incomplete")
            calls = [item for item in payload.get("content", []) if item.get("type") == "tool_use"]
            if len(calls) != 1 or calls[0].get("name") != "submit_dispatch":
                raise ValueError("expected one tool call")
            reply.dispatch = calls[0]["input"]
        else:
            choice = payload["choices"][0]
            if choice.get("finish_reason") != "tool_calls":
                raise ValueError("incomplete")
            calls = choice["message"].get("tool_calls", [])
            if len(calls) != 1 or calls[0]["function"].get("name") != "submit_dispatch":
                raise ValueError("expected one tool call")
            reply.dispatch = parse_json(calls[0]["function"]["arguments"])
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        reply.error = "invalid_or_incomplete_tool_call"
    return reply
