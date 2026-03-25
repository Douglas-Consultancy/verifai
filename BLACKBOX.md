# VerifAI — Project Context

## Project Overview
VerifAI lets you create custom verifiers from a list of rules (principles). It generates synthetic training data using GPT-4o-mini, fine-tunes a small language model (default: TinyLlama-1.1B-Chat), and serves it as a fast API with Prometheus metrics.

## Tech Stack
- **Language:** Python 3.11+
- **API Framework:** FastAPI + Uvicorn
- **ML:** PyTorch, Hugging Face Transformers, PEFT (LoRA fine-tuning), Accelerate
- **Data:** Datasets (Hugging Face), scikit-learn for evaluation metrics
- **External APIs:** OpenAI (synthetic data generation + orchestration)
- **Monitoring:** Prometheus client (metrics at /metrics)
- **Testing:** pytest, httpx (async test client)

## Architecture
1. `train_verifier.py` — Generates synthetic data via OpenAI, fine-tunes model, runs evaluation
2. `serve_verifier.py` — FastAPI server exposing `/verify` endpoint
3. `orchestrate_until_pass.py` — Iterative refinement loop using a larger model + verifier

## Coding Conventions
- Follow PEP 8 style guidelines
- Use type hints on all function signatures
- Write docstrings for all public functions (Google style)
- Use `logging` module (not print statements) for runtime output
- Handle errors with specific exception types
- Keep functions focused and under 50 lines where possible

## Testing
- Use pytest for all tests
- Use httpx.AsyncClient for API endpoint tests
- Test files go in a `tests/` directory
- Name test files `test_<module>.py`
- Include both positive and negative test cases

## Dependencies
- All dependencies are in `requirements.txt`
- Do not add new dependencies without documenting the reason
- Prefer stdlib solutions over new packages

## Security
- Never hardcode API keys or secrets
- All secrets come from environment variables
- The `OPENAI_API_KEY` is required for training and orchestration
- The `VERIFIER_MODEL_PATH` points to the trained model directory
