#!/usr/bin/env python3
"""serve_verifier.py -- Production FastAPI server for a trained VerifAI verifier.

Usage:
    VERIFIER_MODEL_PATH=./my-verifier uvicorn serve_verifier:app --host 0.0.0.0 --port 8000
"""

import json
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel, field_validator
from starlette.responses import Response
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("VERIFIER_MODEL_PATH", "")
MODEL_TYPE = os.environ.get("VERIFIER_MODEL_TYPE", "causal")  # causal | classifier
MAX_NEW_TOKENS = int(os.environ.get("VERIFIER_MAX_NEW_TOKENS", "128"))

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter("verifai_requests_total", "Total verify requests")
REQUEST_LATENCY = Histogram(
    "verifai_request_latency_seconds",
    "Latency of /verify requests",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
VIOLATION_COUNT = Counter("verifai_violations_total", "Total violations detected")
JSON_PARSE_FAILURES = Counter("verifai_json_parse_failures_total", "JSON parse failures")
MODEL_LOADED = Gauge("verifai_model_loaded", "Whether the model is loaded (1=yes, 0=no)")
GPU_MEMORY_MB = Gauge("verifai_gpu_memory_mb", "GPU memory used in MB")
BATCH_REQUEST_COUNT = Counter("verify_batch_requests_total", "Total /verify/batch requests")
BATCH_SIZE = Histogram(
    "verify_batch_size",
    "Number of items per /verify/batch request",
    buckets=(1, 2, 5, 10, 20, 30, 40, 50),
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model = None
tokenizer = None
classifier_pipe = None


def load_model():
    global model, tokenizer, classifier_pipe
    if not MODEL_PATH:
        raise RuntimeError("VERIFIER_MODEL_PATH environment variable is required.")

    if MODEL_TYPE == "classifier":
        classifier_pipe = pipeline("text-classification", model=MODEL_PATH, device_map="auto")
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float32, device_map="auto"
        )
        model.eval()

    MODEL_LOADED.set(1)
    print(f"Model loaded from {MODEL_PATH} (type={MODEL_TYPE})")


def update_gpu_metrics():
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e6
        GPU_MEMORY_MB.set(round(mem, 1))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    MODEL_LOADED.set(0)


app = FastAPI(title="VerifAI", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class VerifyRequest(BaseModel):
    principles: list[str]
    response: str


class Verdict(BaseModel):
    violations: list[str]
    confidence: float


class VerifyResponse(BaseModel):
    verdict: Verdict
    latency_ms: float


class BatchItem(BaseModel):
    principles: list[str]
    response: str


class BatchResult(BaseModel):
    violations: list[str]
    confidence: float


class BatchRequest(BaseModel):
    items: list[BatchItem]

    @field_validator("items")
    @classmethod
    def validate_items(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("items must not be empty")
        if len(v) > 50:
            raise ValueError("items must not exceed 50")
        return v


class BatchResponse(BaseModel):
    results: list[BatchResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest):
    REQUEST_COUNT.inc()
    start = time.perf_counter()

    all_violations: list[str] = []
    total_confidence = 0.0

    for principle in req.principles:
        if MODEL_TYPE == "classifier":
            result = classifier_pipe(f"{principle} ||| {req.response}")[0]
            is_violation = result["label"].lower() in ("fail", "violation", "1")
            conf = result["score"]
        else:
            prompt = (
                f"<|user|>\n"
                f"Verify the following response against the principle: \"{principle}\"\n\n"
                f"Response: \"{req.response}\"\n"
                f"<|assistant|>\n"
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(next(model.parameters()).device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
                )
            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            try:
                verdict = json.loads(generated)
                is_violation = bool(verdict.get("violations"))
                conf = float(verdict.get("confidence", 0.5))
            except (json.JSONDecodeError, ValueError):
                JSON_PARSE_FAILURES.inc()
                is_violation = True
                conf = 0.5

        if is_violation:
            all_violations.append(principle)
            VIOLATION_COUNT.inc()
        total_confidence += conf

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    avg_confidence = round(total_confidence / max(len(req.principles), 1), 2)

    update_gpu_metrics()

    return VerifyResponse(
        verdict=Verdict(violations=all_violations, confidence=avg_confidence),
        latency_ms=elapsed_ms,
    )


def _run_inference(principles: list[str], response: str) -> BatchResult:
    """Run model inference for a single (principles, response) pair and return a BatchResult."""
    all_violations: list[str] = []
    total_confidence = 0.0

    for principle in principles:
        if MODEL_TYPE == "classifier":
            result = classifier_pipe(f"{principle} ||| {response}")[0]
            is_violation = result["label"].lower() in ("fail", "violation", "1")
            conf = result["score"]
        else:
            prompt = (
                f"<|user|>\n"
                f"Verify the following response against the principle: \"{principle}\"\n\n"
                f"Response: \"{response}\"\n"
                f"<|assistant|>\n"
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(next(model.parameters()).device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
                )
            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            try:
                verdict = json.loads(generated)
                is_violation = bool(verdict.get("violations"))
                conf = float(verdict.get("confidence", 0.5))
            except (json.JSONDecodeError, ValueError):
                JSON_PARSE_FAILURES.inc()
                is_violation = True
                conf = 0.5

        if is_violation:
            all_violations.append(principle)
            VIOLATION_COUNT.inc()
        total_confidence += conf

    avg_confidence = round(total_confidence / max(len(principles), 1), 2)
    return BatchResult(violations=all_violations, confidence=avg_confidence)


@app.post("/verify/batch", response_model=BatchResponse)
async def verify_batch(req: BatchRequest):
    BATCH_REQUEST_COUNT.inc()
    BATCH_SIZE.observe(len(req.items))

    results = [
        _run_inference(item.principles, item.response) for item in req.items
    ]
    update_gpu_metrics()
    return BatchResponse(results=results)


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    loaded = MODEL_LOADED._value.get()
    if not loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}
