# VerifAI — Copilot Instructions

## Project Overview
VerifAI is a custom AI verifier framework. It generates synthetic training data from user-defined rules (principles), fine-tunes a small language model (TinyLlama-1.1B-Chat by default), and serves verdicts via a FastAPI endpoint with Prometheus metrics.

## Tech Stack
- **Language:** Python 3.10+
- **ML Framework:** Hugging Face Transformers, PEFT/LoRA for fine-tuning
- **API:** FastAPI with Uvicorn
- **Monitoring:** Prometheus metrics at `/metrics`, Grafana dashboards
- **External APIs:** OpenAI (GPT-4o-mini for data generation, GPT-4 for orchestration)
- **Containerization:** Docker Compose for monitoring stack

## Key Files
- `train_verifier.py` — Synthetic data generation + fine-tuning pipeline
- `serve_verifier.py` — Production FastAPI server with `/verify` endpoint
- `orchestrate_until_pass.py` — Iterative refinement loop using a frontier model + verifier
- `principles.txt` — User-defined rules the verifier checks against
- `monitoring/` — Prometheus, Grafana, and alerting configs

## Coding Standards
- Use type hints on all function signatures
- JSON output from the verifier must always include `violations` (list of strings) and `confidence` (float 0-1)
- Environment variables for configuration (see README for full list)
- All new endpoints must expose Prometheus counters/histograms
- Tests should cover both causal and classifier model types
- Keep inference latency under 100ms on CPU as a design target

## Build & Test
```bash
pip install -r requirements.txt
python train_verifier.py --principles principles.txt --output-dir ./my-verifier
python -m pytest tests/
```

## Common Patterns
- Verifier output is always JSON: `{"verdict": {"violations": [...], "confidence": 0.XX}, "latency_ms": NN}`
- The orchestrator loops: generate draft → verify → fix → re-verify until pass or max_iters
- Principles are plain text, one per line
- Model supports two architectures: `causal` (default, generates JSON) and `classifier` (binary pass/fail)
