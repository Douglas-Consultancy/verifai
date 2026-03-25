"""Unit tests for serve_verifier.py – batch endpoint."""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def _make_client(model_type: str = "classifier"):
    """Yield a (TestClient, mock_pipe) pair with load_model stubbed out.

    classifier_pipe is patched to return a *pass* verdict by default so that
    individual tests can override it for violation scenarios.
    """
    mock_pipe = MagicMock(return_value=[{"label": "pass", "score": 0.9}])

    with (
        patch("serve_verifier.load_model"),
        patch("serve_verifier.MODEL_TYPE", model_type),
        patch("serve_verifier.classifier_pipe", mock_pipe),
        patch("serve_verifier.MODEL_LOADED") as mock_loaded,
    ):
        mock_loaded._value.get.return_value = 1

        import serve_verifier  # noqa: PLC0415 – intentional late import

        with TestClient(serve_verifier.app) as client:
            yield client, mock_pipe


# ---------------------------------------------------------------------------
# /verify/batch – happy-path tests
# ---------------------------------------------------------------------------

class TestVerifyBatchHappyPath:
    def test_single_item_no_violations(self):
        with _make_client() as (client, mock_pipe):
            mock_pipe.return_value = [{"label": "pass", "score": 0.95}]
            resp = client.post(
                "/verify/batch",
                json={
                    "items": [
                        {"principles": ["Be concise"], "response": "Short answer."}
                    ]
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert len(body["results"]) == 1
        result = body["results"][0]
        assert result["violations"] == []
        assert 0.0 <= result["confidence"] <= 1.0

    def test_single_item_with_violation(self):
        with _make_client() as (client, mock_pipe):
            mock_pipe.return_value = [{"label": "violation", "score": 0.88}]
            resp = client.post(
                "/verify/batch",
                json={
                    "items": [
                        {
                            "principles": ["Never make promises"],
                            "response": "I guarantee results.",
                        }
                    ]
                },
            )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert "Never make promises" in result["violations"]
        assert result["confidence"] == pytest.approx(0.88, abs=0.01)

    def test_multiple_items_returns_matching_count(self):
        with _make_client() as (client, mock_pipe):
            mock_pipe.return_value = [{"label": "pass", "score": 0.9}]
            items = [
                {"principles": ["Be honest"], "response": f"response {i}"}
                for i in range(5)
            ]
            resp = client.post("/verify/batch", json={"items": items})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 5

    def test_exactly_50_items_accepted(self):
        with _make_client() as (client, mock_pipe):
            mock_pipe.return_value = [{"label": "pass", "score": 0.9}]
            items = [
                {"principles": ["Be concise"], "response": f"r{i}"}
                for i in range(50)
            ]
            resp = client.post("/verify/batch", json={"items": items})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 50

    def test_item_with_multiple_principles(self):
        """Each principle is checked; violating ones appear in violations list."""
        call_results = [
            [{"label": "violation", "score": 0.9}],  # principle 1 – violates
            [{"label": "pass", "score": 0.8}],       # principle 2 – passes
        ]
        with _make_client() as (client, mock_pipe):
            mock_pipe.side_effect = call_results
            resp = client.post(
                "/verify/batch",
                json={
                    "items": [
                        {
                            "principles": ["No promises", "Be polite"],
                            "response": "I guarantee you.",
                        }
                    ]
                },
            )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["violations"] == ["No promises"]
        assert len(result["violations"]) == 1


# ---------------------------------------------------------------------------
# /verify/batch – validation error tests
# ---------------------------------------------------------------------------

class TestVerifyBatchValidation:
    def test_empty_items_returns_422(self):
        with _make_client() as (client, _):
            resp = client.post("/verify/batch", json={"items": []})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any("empty" in str(d).lower() for d in detail)

    def test_51_items_returns_422(self):
        with _make_client() as (client, _):
            items = [
                {"principles": ["p"], "response": "r"} for _ in range(51)
            ]
            resp = client.post("/verify/batch", json={"items": items})
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any("50" in str(d) for d in detail)

    def test_missing_items_field_returns_422(self):
        with _make_client() as (client, _):
            resp = client.post("/verify/batch", json={})
        assert resp.status_code == 422

    def test_malformed_item_returns_422(self):
        with _make_client() as (client, _):
            resp = client.post(
                "/verify/batch",
                json={"items": [{"principles": "not-a-list", "response": "r"}]},
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /verify/batch – response schema tests
# ---------------------------------------------------------------------------

class TestVerifyBatchResponseSchema:
    def test_result_has_violations_and_confidence(self):
        with _make_client() as (client, mock_pipe):
            mock_pipe.return_value = [{"label": "pass", "score": 0.75}]
            resp = client.post(
                "/verify/batch",
                json={"items": [{"principles": ["Be honest"], "response": "Yes."}]},
            )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert "violations" in result
        assert "confidence" in result
        assert isinstance(result["violations"], list)
        assert isinstance(result["confidence"], float)

"""Unit / integration tests for serve_verifier.py using FastAPI TestClient.

All model loading and inference is mocked via the ``test_client`` fixture
defined in ``conftest.py``.  No real model files, GPU, or network calls are
made during these tests.
"""

import pytest


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_returns_200(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


def test_metrics_returns_200(test_client):
    response = test_client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_metrics_contains_prometheus_text(test_client):
    response = test_client.get("/metrics")
    # Prometheus text format always starts with a "# HELP" or "# TYPE" comment
    assert b"# HELP" in response.content or b"# TYPE" in response.content


# ---------------------------------------------------------------------------
# POST /verify — happy path
# ---------------------------------------------------------------------------


def test_verify_valid_payload_returns_200(test_client, mock_principles):
    response = test_client.post(
        "/verify",
        json={"principles": mock_principles, "response": "Hello, how can I help?"},
    )
    assert response.status_code == 200


def test_verify_response_has_verdict_violations(test_client, mock_principles):
    response = test_client.post(
        "/verify",
        json={"principles": mock_principles, "response": "A polite, concise reply."},
    )
    data = response.json()
    assert "verdict" in data
    assert "violations" in data["verdict"]
    assert isinstance(data["verdict"]["violations"], list)


def test_verify_response_has_verdict_confidence(test_client, mock_principles):
    response = test_client.post(
        "/verify",
        json={"principles": mock_principles, "response": "A polite, concise reply."},
    )
    data = response.json()
    assert "confidence" in data["verdict"]
    assert isinstance(data["verdict"]["confidence"], float)


def test_verify_response_has_latency_ms(test_client, mock_principles):
    response = test_client.post(
        "/verify",
        json={"principles": mock_principles, "response": "A polite, concise reply."},
    )
    data = response.json()
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], float)


# ---------------------------------------------------------------------------
# POST /verify — validation errors (422)
# ---------------------------------------------------------------------------


def test_verify_missing_response_field_returns_422(test_client, mock_principles):
    response = test_client.post(
        "/verify",
        json={"principles": mock_principles},
    )
    assert response.status_code == 422


def test_verify_missing_principles_field_returns_422(test_client):
    response = test_client.post(
        "/verify",
        json={"response": "Hello"},
    )
    assert response.status_code == 422


def test_verify_empty_body_returns_422(test_client):
    response = test_client.post("/verify", json={})
    assert response.status_code == 422
