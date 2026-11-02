import json

import pytest

from aeolus.agents.adapters import AgentSpec
from aeolus.agents.cli_adapters import decode_cli

DISPATCH = {"decision_id": "example:0", "assignments": [], "rationale": "", "memory": ""}


def stream(*events):
    return "\n".join(json.dumps(event) for event in events)


def codex_events(extra=None):
    return stream(
        {"type": "thread.started", "thread_id": "id-12345678"},
        extra or {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(DISPATCH)}},
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 120, "cached_input_tokens": 30, "output_tokens": 40},
        },
    )


def test_codex_preserves_usage_and_reported_model_without_inventing_identity():
    reply = decode_cli("codex_cli", codex_events(), "model: gpt-example\n", 0)
    assert not reply.error
    assert reply.dispatch == DISPATCH
    assert reply.usage == {"input_tokens": 120, "output_tokens": 40}
    assert reply.model == "gpt-example"
    assert reply.metadata["cached_input_tokens"] == 30
    assert decode_cli("codex_cli", codex_events(), "", 0).model is None


def test_codex_tool_use_makes_an_otherwise_valid_dispatch_ineligible():
    tool = {"type": "item.completed", "item": {"type": "command_execution", "command": "cat hidden"}}
    reply = decode_cli("codex_cli", codex_events(tool), "", 0)
    assert reply.error == "cli_disallowed_tool_use"
    assert reply.metadata["tools_used"] == ["command_execution"]


def test_claude_uses_structured_result_and_counts_cache_tokens():
    events = stream(
        {"type": "system", "subtype": "init", "model": "claude-example", "claude_code_version": "1.2"},
        {
            "type": "assistant",
            "message": {
                "model": "claude-example",
                "content": [{"type": "tool_use", "name": "StructuredOutput", "input": DISPATCH}],
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "structured_output": DISPATCH,
            "usage": {
                "input_tokens": 2,
                "cache_creation_input_tokens": 80,
                "cache_read_input_tokens": 20,
                "output_tokens": 30,
            },
            "session_id": "session",
            "modelUsage": {"claude-example": {}},
        },
    )
    reply = decode_cli("claude_cli", events, "", 0)
    assert not reply.error
    assert reply.usage == {"input_tokens": 102, "output_tokens": 30}
    assert reply.dispatch == DISPATCH
    assert reply.model == "claude-example"
    assert reply.metadata["tools_used"] == []


def test_claude_missing_usage_stays_unknown_and_tool_use_fails():
    events = stream(
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read"}]}},
        {"type": "result", "subtype": "success", "structured_output": DISPATCH},
    )
    reply = decode_cli("claude_cli", events, "", 0)
    assert reply.usage == {"input_tokens": None, "output_tokens": None}
    assert reply.error == "cli_disallowed_tool_use"


@pytest.mark.parametrize("provider", ["codex_cli", "claude_cli"])
def test_cli_rejects_nonzero_exit_even_with_an_answer(provider):
    events = (
        codex_events()
        if provider == "codex_cli"
        else stream({"type": "result", "subtype": "success", "structured_output": DISPATCH})
    )
    assert decode_cli(provider, events, "", 1).error == "cli_request_failed"


def test_cli_requires_honest_budget_and_explicit_reasoning_effort():
    with pytest.raises(ValueError, match="max_output_tokens"):
        AgentSpec("cli", provider="codex_cli", model="gpt-example", reasoning_effort="high")
    with pytest.raises(ValueError, match="reasoning_effort"):
        AgentSpec("cli", provider="codex_cli", model="gpt-example", max_output_tokens=None)
    AgentSpec(
        "cli", provider="codex_cli", model="gpt-example", max_output_tokens=None, reasoning_effort="high"
    )


def test_cli_does_not_repair_malformed_or_duplicate_answers():
    assert decode_cli("codex_cli", "not json", "", 0).error == "cli_invalid_event_stream"
    extra = {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(DISPATCH)}}
    assert decode_cli("codex_cli", codex_events(extra), "", 0).error == "cli_incomplete_or_multiple_answers"


def test_authentication_failure_is_not_mislabeled_as_malformed_dispatch():
    events = stream(
        {"type": "assistant", "message": {"model": "<synthetic>", "content": []}},
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": "Failed to authenticate: OAuth session expired and could not be refreshed",
        },
    )
    reply = decode_cli("claude_cli", events, "", 1)
    assert reply.error == "cli_authentication_failed"
    assert reply.model is None


def test_codex_bridge_provenance_identifies_the_actual_session_configuration():
    event = {
        "type": "bridge.provenance",
        "model": "gpt-example",
        "reasoning_effort": "high",
        "source": "codex_turn_context",
        "context_sha256": "digest",
    }
    reply = decode_cli("codex_cli", codex_events(event), "", 0)
    assert reply.model == "gpt-example"
    assert reply.metadata["model_identity_source"] == "codex_turn_context"
    assert reply.metadata["reasoning_effort"] == "high"
