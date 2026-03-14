# syntax=docker/dockerfile:1

# ---- builder: install Python dependencies ----
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- runner: minimal production image ----
FROM python:3.11-slim AS runner

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy only the application code needed at runtime
COPY serve_verifier.py .

EXPOSE 8000

# Model path is supplied at runtime via -e VERIFIER_MODEL_PATH=...
ENV VERIFIER_MODEL_PATH=""

CMD ["uvicorn", "serve_verifier:app", "--host", "0.0.0.0", "--port", "8000"]
