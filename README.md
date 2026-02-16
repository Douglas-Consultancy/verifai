# VerifAI – Build Your Own Custom Verifier in Minutes

**VerifAI** lets you create a **custom verifier** from just a list of rules (principles).
It generates synthetic training data using GPT-4o-mini, fine-tunes a small language model,
and serves it as a fast API with Prometheus metrics and a Grafana dashboard.

## How It Works

1. **Synthetic data generation** – GPT-4o-mini creates realistic responses that either follow or violate each principle.
2. **Fine-tuning** – A small instruction-tuned causal LM (default: TinyLlama-1.1B-Chat) learns to output JSON verdicts.
3. **Functional evaluation** – After training, the model generates verdicts on a held-out set and we compute accuracy, precision, recall, and F1.
4. **Inference** – The model runs in milliseconds on CPU or GPU, emitting violations and confidence as JSON.
5. **Orchestration** – A larger model (e.g., GPT-4) uses the verifier's output to iteratively improve drafts until they pass.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

### 2. Create your principles file

```
Never make promises you cannot keep.
Cite sources if you reference data.
Be concise and avoid jargon.
```

### 3. Train your verifier

```bash
python train_verifier.py --principles principles.txt --output-dir ./my-verifier
```

This generates ~150-300 examples (depending on principle count), fine-tunes a model,
runs functional evaluation, and saves everything to `./my-verifier`.

Check `./my-verifier/eval_metrics.json` for accuracy, precision, recall, and F1.

### 4. Serve the verifier

```bash
export VERIFIER_MODEL_PATH=./my-verifier
uvicorn serve_verifier:app --host 0.0.0.0 --port 8000
```

### 5. Test it

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "principles": ["Never make promises you cannot keep"],
    "response": "I guarantee you will double your money."
  }'
```

Response:

```json
{
  "verdict": {
    "violations": ["Never make promises you cannot keep"],
    "confidence": 0.92
  },
  "latency_ms": 45
}
```

### 6. Use the orchestrator to refine drafts

```bash
export VERIFIER_URL=http://localhost:8000/verify
python orchestrate_until_pass.py \
  --task "Write a customer email about a delayed shipment" \
  --principles principles.txt \
  --max_iters 3 \
  --output trace.jsonl \
  --print_trace
```

### 7. One-command quickstart

```bash
chmod +x run_verifai.sh
./run_verifai.sh
```

## Monitoring

The service exposes `/metrics` in Prometheus format. A pre-built Grafana dashboard
and alerting rules are in `monitoring/`. Run the full stack with Docker Compose:

```bash
cd monitoring
docker-compose -f docker-compose-grafana.yml up -d
```

Then open `http://localhost:3000` (admin/admin) to see live dashboards.

### Alerts included

| Alert | Severity | Condition |
|-------|----------|-----------|
| High p95 latency | critical | > 1s for 2 min |
| JSON parse failures | critical | > 0.1/s for 1 min |
| High violation rate | warning | > 10/s for 3 min |
| GPU memory pressure | warning | > 6 GB for 5 min |
| Model unloaded | critical | model_loaded == 0 for 1 min |

## Model types

VerifAI supports two model architectures via `VERIFIER_MODEL_TYPE`:

- `causal` (default) – instruction-tuned causal LM that generates JSON verdicts. This is what `train_verifier.py` produces.
- `classifier` – binary sequence classification model (pass/fail). Bring your own model.

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `VERIFIER_MODEL_PATH` | (required) | Path to trained model directory |
| `VERIFIER_MODEL_TYPE` | `causal` | `causal` or `classifier` |
| `VERIFIER_MAX_NEW_TOKENS` | `128` | Max tokens to generate (causal only) |
| `OPENAI_API_KEY` | (required for training/orchestration) | OpenAI API key |
| `BIG_MODEL` | `gpt-4.1` | Frontier model for orchestrator |
| `VERIFIER_URL` | `http://localhost:8000/verify` | Verifier endpoint for orchestrator |

## Project structure

```
verifai/
├── train_verifier.py          # Train from principles
├── serve_verifier.py          # Production API server
├── orchestrate_until_pass.py  # Iterative refinement loop
├── requirements.txt
├── principles.txt             # Example principles
├── run_verifai.sh             # One-command quickstart
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── .github/workflows/test.yml
├── monitoring/
│   ├── docker-compose-grafana.yml
│   ├── prometheus.yml
│   ├── alert-rules.yml
│   ├── alertmanager.yml
│   ├── grafana-dashboard.json
│   └── grafana-provisioning/
│       ├── datasources/datasource.yml
│       └── dashboards/dashboard.yml
└── assets/
    ├── logo_prompt.txt
    ├── demo_video_script.md
    └── github_description.txt
```

## License

Apache 2.0 – use freely, even commercially.
