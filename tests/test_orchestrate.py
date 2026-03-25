"""Unit tests for orchestrate_until_pass.py.

All external calls (OpenAI API, verifier HTTP endpoint) are mocked so that
no network traffic is produced during the test run.
"""

import json

import pytest
from unittest.mock import MagicMock, patch

import orchestrate_until_pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_openai_client(content: str = "Draft response") -> MagicMock:
    """Return a mock OpenAI client whose completions return *content*."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    client = MagicMock()
    client.chat.completions.create.return_value = mock_resp
    return client


def _verifier_result(violations: list | None = None, confidence: float = 0.9) -> dict:
    """Build a verifier API response dict."""
    return {
        "verdict": {
            "violations": violations if violations is not None else [],
            "confidence": confidence,
        },
        "latency_ms": 5.0,
    }


# ---------------------------------------------------------------------------
# orchestrate() — early stop on pass
# ---------------------------------------------------------------------------


def test_orchestrate_stops_early_when_no_violations(mock_principles):
    client = _mock_openai_client()
    with patch(
        "orchestrate_until_pass.call_verifier",
        return_value=_verifier_result(),
    ):
        trace = orchestrate_until_pass.orchestrate("task", mock_principles, 5, client)

    assert len(trace) == 1
    assert trace[0]["passed"] is True
    assert trace[0]["violations"] == []


def test_orchestrate_draft_is_called_once_on_early_pass(mock_principles):
    client = _mock_openai_client()
    with patch(
        "orchestrate_until_pass.call_verifier",
        return_value=_verifier_result(),
    ):
        orchestrate_until_pass.orchestrate("task", mock_principles, 5, client)

    assert client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# orchestrate() — retries up to max_iters
# ---------------------------------------------------------------------------


def test_orchestrate_retries_up_to_max_iters(mock_principles):
    client = _mock_openai_client()
    with patch(
        "orchestrate_until_pass.call_verifier",
        return_value=_verifier_result(violations=["Be polite"]),
    ):
        trace = orchestrate_until_pass.orchestrate("task", mock_principles, 3, client)

    assert len(trace) == 3
    assert all(not step["passed"] for step in trace)


def test_orchestrate_generates_draft_each_iteration(mock_principles):
    client = _mock_openai_client()
    with patch(
        "orchestrate_until_pass.call_verifier",
        return_value=_verifier_result(violations=["Be polite"]),
    ):
        orchestrate_until_pass.orchestrate("task", mock_principles, 3, client)

    assert client.chat.completions.create.call_count == 3


def test_orchestrate_stops_as_soon_as_pass(mock_principles):
    """Verify pass on iteration 2 of 5 produces a trace of length 2."""
    client = _mock_openai_client()
    side_effects = [
        _verifier_result(violations=["Be polite"]),
        _verifier_result(violations=[]),
    ]
    with patch(
        "orchestrate_until_pass.call_verifier",
        side_effect=side_effects,
    ):
        trace = orchestrate_until_pass.orchestrate("task", mock_principles, 5, client)

    assert len(trace) == 2
    assert trace[-1]["passed"] is True


# ---------------------------------------------------------------------------
# orchestrate() — trace structure
# ---------------------------------------------------------------------------


def test_orchestrate_trace_contains_expected_fields(mock_principles):
    client = _mock_openai_client(content="A great response")
    with patch(
        "orchestrate_until_pass.call_verifier",
        return_value=_verifier_result(),
    ):
        trace = orchestrate_until_pass.orchestrate("task", mock_principles, 3, client)

    step = trace[0]
    for key in ("iteration", "draft", "violations", "confidence", "latency_ms", "passed"):
        assert key in step, f"Missing key: {key}"


def test_orchestrate_trace_draft_matches_mock(mock_principles):
    client = _mock_openai_client(content="My specific draft")
    with patch(
        "orchestrate_until_pass.call_verifier",
        return_value=_verifier_result(),
    ):
        trace = orchestrate_until_pass.orchestrate("task", mock_principles, 3, client)

    assert trace[0]["draft"] == "My specific draft"


def test_orchestrate_trace_iteration_numbers(mock_principles):
    client = _mock_openai_client()
    with patch(
        "orchestrate_until_pass.call_verifier",
        return_value=_verifier_result(violations=["Be concise"]),
    ):
        trace = orchestrate_until_pass.orchestrate("task", mock_principles, 3, client)

    for i, step in enumerate(trace, start=1):
        assert step["iteration"] == i


# ---------------------------------------------------------------------------
# trace.jsonl output
# ---------------------------------------------------------------------------


def test_orchestrate_trace_jsonl_output(mock_principles, tmp_path):
    """Verify that trace steps serialize to valid JSONL."""
    client = _mock_openai_client()
    output_file = tmp_path / "trace.jsonl"

    with patch(
        "orchestrate_until_pass.call_verifier",
        return_value=_verifier_result(violations=["Be polite"]),
    ):
        trace = orchestrate_until_pass.orchestrate("task", mock_principles, 2, client)

    with open(output_file, "w") as f:
        for step in trace:
            f.write(json.dumps(step) + "\n")

    lines = output_file.read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        data = json.loads(line)
        for key in ("iteration", "draft", "violations", "passed"):
            assert key in data


def test_orchestrate_trace_jsonl_single_pass(mock_principles, tmp_path):
    """A passing run produces exactly one line in the JSONL file."""
    client = _mock_openai_client()
    output_file = tmp_path / "trace_pass.jsonl"

    with patch(
        "orchestrate_until_pass.call_verifier",
        return_value=_verifier_result(),
    ):
        trace = orchestrate_until_pass.orchestrate("task", mock_principles, 5, client)

    with open(output_file, "w") as f:
        for step in trace:
            f.write(json.dumps(step) + "\n")

    lines = output_file.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["passed"] is True


# ---------------------------------------------------------------------------
# call_verifier()
# ---------------------------------------------------------------------------


def test_call_verifier_sends_correct_payload(mock_principles, mock_verifier_response):
    mock_http_resp = MagicMock()
    mock_http_resp.json.return_value = mock_verifier_response

    with patch(
        "orchestrate_until_pass.requests.post", return_value=mock_http_resp
    ) as mock_post:
        result = orchestrate_until_pass.call_verifier("test response", mock_principles)

    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {
        "principles": mock_principles,
        "response": "test response",
    }
    assert result == mock_verifier_response


def test_call_verifier_returns_parsed_json(mock_principles, mock_verifier_response):
    mock_http_resp = MagicMock()
    mock_http_resp.json.return_value = mock_verifier_response

    with patch("orchestrate_until_pass.requests.post", return_value=mock_http_resp):
        result = orchestrate_until_pass.call_verifier("test response", mock_principles)

    assert result["verdict"]["violations"] == []
    assert result["verdict"]["confidence"] == 0.95
