from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from aeolus.agents.adapters import AgentAdapter, AgentFailure, AgentSpec, baseline_dispatch, decode_reply
from aeolus.agents.desk import IncidentDesk
from aeolus.config import ScenarioConfig


@pytest.fixture
def observation():
    return IncidentDesk(ScenarioConfig(width=32, height=32, max_tasks=24), 17).observation()


def envelope(provider, dispatch):
    common = {"model": "fixture-model", "id": "fixture-id", "usage": {"input_tokens": 12, "output_tokens": 7}}
    if provider == "openai":
        return {
            **common,
            "status": "completed",
            "output": [
                {"type": "function_call", "name": "submit_dispatch", "arguments": json.dumps(dispatch)}
            ],
        }
    if provider == "anthropic":
        return {
            **common,
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "name": "submit_dispatch", "input": dispatch}],
        }
    return {
        **common,
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {"function": {"name": "submit_dispatch", "arguments": json.dumps(dispatch)}}
                    ]
                },
            }
        ],
    }


@pytest.mark.parametrize(
    "provider,route",
    [("openai", "/v1/responses"), ("anthropic", "/v1/messages"), ("compatible", "/v1/chat/completions")],
)
def test_adapters_over_real_http_fixture(observation, provider, route, monkeypatch):
    dispatch = baseline_dispatch(observation, "greedy")
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, body, dict(self.headers)))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(envelope(provider, dispatch)).encode())

        def log_message(self, *args):
            pass

    monkeypatch.setenv("DESK_FIXTURE_KEY", "fixture-secret")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        spec = AgentSpec(
            "fixture",
            provider,
            "fixture-model",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            api_key_env="DESK_FIXTURE_KEY",
            max_output_tokens=100,
        )
        result = AgentAdapter(spec).act(observation)
        assert result.dispatch == dispatch
        assert result.usage == {"input_tokens": 12, "output_tokens": 7}
        assert result.response_id == "fixture-id"
        path, body, headers = received[0]
        assert path == route
        assert "fixture-secret" not in json.dumps(body)
        assert "fixture-secret" not in json.dumps(spec.public_dict())
        if provider == "openai":
            assert body["store"] is False
            assert body["tools"][0]["strict"] is True
            assert body["max_output_tokens"] == 100
        else:
            assert body["max_tokens"] == 100
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "compatible"])
def test_incomplete_or_wrong_tool_fails_with_usage_retained(provider):
    payload = envelope(provider, {})
    if provider == "openai":
        payload["status"] = "incomplete"
    elif provider == "anthropic":
        payload["stop_reason"] = "max_tokens"
    else:
        payload["choices"][0]["finish_reason"] = "length"
    reply = decode_reply(provider, payload)
    assert reply.error
    assert reply.usage["output_tokens"] == 7
    payload = envelope(provider, {})
    if provider == "openai":
        payload["output"][0]["name"] = "wrong_tool"
    elif provider == "anthropic":
        payload["content"][0]["name"] = "wrong_tool"
    else:
        payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = "wrong_tool"
    assert decode_reply(provider, payload).error


def test_command_protocol_and_failures(observation, tmp_path):
    script = tmp_path / "agent.py"
    script.write_text(
        "import json,sys\nr=json.load(sys.stdin)\n"
        "from aeolus.agents.adapters import baseline_dispatch\n"
        "print(json.dumps(baseline_dispatch(r['observation'], 'greedy')))\n"
    )
    spec = AgentSpec("external", "command", "fixture", command=(sys.executable, str(script)))
    assert AgentAdapter(spec).act(observation).dispatch == baseline_dispatch(observation, "greedy")
    timeout = AgentSpec(
        "timeout",
        "command",
        "fixture",
        timeout_s=0.05,
        command=(sys.executable, "-c", "import time; time.sleep(10)"),
    )
    with pytest.raises(AgentFailure, match="command_timeout"):
        AgentAdapter(timeout).act(observation)
    invalid = AgentSpec("invalid", "command", "fixture", command=(sys.executable, "-c", "print('not-json')"))
    with pytest.raises(AgentFailure, match="invalid_json"):
        AgentAdapter(invalid).act(observation)


def test_missing_key_fails_before_request(monkeypatch):
    monkeypatch.delenv("DESK_UNSET_KEY", raising=False)
    with pytest.raises(ValueError, match="DESK_UNSET_KEY"):
        AgentAdapter(AgentSpec("missing", "openai", "model", api_key_env="DESK_UNSET_KEY"))


@pytest.mark.parametrize("usage", [[], "bad", {"input_tokens": "12", "output_tokens": -1}])
def test_malformed_usage_is_unknown_not_zero_or_a_crash(usage):
    payload = envelope("anthropic", {})
    payload["usage"] = usage
    reply = decode_reply("anthropic", payload)
    assert reply.usage == {"input_tokens": None, "output_tokens": None}


def test_anthropic_cache_tokens_are_counted():
    payload = envelope("anthropic", {})
    payload["usage"].update(cache_creation_input_tokens=20, cache_read_input_tokens=100)
    assert decode_reply("anthropic", payload).usage["input_tokens"] == 132


@pytest.mark.parametrize("provider", ["openai", "anthropic", "compatible"])
def test_multiple_tool_calls_are_never_silently_discarded(provider):
    payload = envelope(provider, {})
    if provider == "openai":
        payload["output"] *= 2
    elif provider == "anthropic":
        payload["content"] *= 2
    else:
        payload["choices"][0]["message"]["tool_calls"] *= 2
    assert decode_reply(provider, payload).error


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.invalid/v1",
        "https://secret@host/v1",
        "https://host/v1?key=secret",
        "https://host/v1#secret",
    ],
)
def test_credentials_and_insecure_remote_urls_are_rejected(url):
    with pytest.raises(ValueError):
        AgentSpec("bad", "compatible", "model", base_url=url)
