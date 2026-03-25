"""Shared pytest fixtures for VerifAI test suite."""

import json

import pytest
from starlette.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_principles():
    """A small list of sample principles for use in tests."""
    return ["Be polite", "Be concise"]


@pytest.fixture
def mock_verifier_response():
    """A passing verifier response payload (no violations)."""
    return {
        "verdict": {"violations": [], "confidence": 0.95},
        "latency_ms": 12.5,
    }


@pytest.fixture
def test_client():
    """FastAPI TestClient with all model loading fully mocked.

    The fixture patches ``serve_verifier.load_model`` so that no real
    model or tokenizer is ever loaded during tests.  Mock objects for
    ``tokenizer`` and ``model`` are installed on the module so that the
    ``/verify`` endpoint can exercise its inference path without touching
    the file-system or GPU.
    """
    import serve_verifier

    # ------------------------------------------------------------------
    # Build a fake tokenizer whose return value survives **-unpacking
    # inside the endpoint (``model.generate(**inputs, ...)``).
    # ------------------------------------------------------------------
    mock_input_ids = MagicMock()
    mock_input_ids.shape = [1, 10]  # shape[1] used as slice start
    inputs_dict = {"input_ids": mock_input_ids}

    tokenizer_call_result = MagicMock()
    tokenizer_call_result.to.return_value = inputs_dict  # .to(device) → real dict

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "<pad>"
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer.return_value = tokenizer_call_result
    mock_tokenizer.decode.return_value = json.dumps(
        {"violations": [], "confidence": 0.9}
    )

    # ------------------------------------------------------------------
    # Build a fake model.
    # Use side_effect so each call to parameters() returns a fresh
    # iterator (next() consumes it).
    # ------------------------------------------------------------------
    mock_param = MagicMock()
    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([mock_param])

    def _fake_load():
        serve_verifier.tokenizer = mock_tokenizer
        serve_verifier.model = mock_model
        serve_verifier.MODEL_LOADED.set(1)

    with patch("serve_verifier.load_model", side_effect=_fake_load):
        with TestClient(serve_verifier.app) as client:
            yield client
