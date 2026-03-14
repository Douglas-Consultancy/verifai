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
