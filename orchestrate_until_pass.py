#!/usr/bin/env python3
"""orchestrate_until_pass.py -- Iterative refinement loop using a frontier model + VerifAI verifier.

Usage:
    VERIFIER_URL=http://localhost:8000/verify python orchestrate_until_pass.py \
        --task "Write a customer email about a delayed shipment" \
        --principles principles.txt \
        --max_iters 3 \
        --output trace.jsonl \
        --print_trace
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
VERIFIER_URL = os.environ.get("VERIFIER_URL", "http://localhost:8000/verify")


def call_verifier(response_text: str, principles: list[str]) -> dict:
    """Call the VerifAI verifier API."""
    resp = requests.post(
        VERIFIER_URL,
        json={"principles": principles, "response": response_text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def generate_draft(client: OpenAI, task: str, principles: list[str], feedback: str | None = None) -> str:
    """Generate or refine a draft using the frontier model."""
    system_msg = (
        "You are a helpful assistant. Follow ALL of the user's principles strictly.\n\n"
        "Principles:\n" + "\n".join(f"- {p}" for p in principles)
    )
    messages = [{"role": "system", "content": system_msg}]

    if feedback:
        messages.append({"role": "user", "content": (
            f"Task: {task}\n\n"
            f"Your previous draft violated these principles: {feedback}\n\n"
            f"Please rewrite the response to fix ALL violations while still completing the task."
        )})
    else:
        messages.append({"role": "user", "content": task})

    resp = client.chat.completions.create(
        model=BIG_MODEL,
        messages=messages,
        temperature=0.7,
    )
    return resp.choices[0].message.content


def orchestrate(task: str, principles: list[str], max_iters: int, client: OpenAI) -> list[dict]:
    """Run the generate-verify-refine loop. Returns the full trace."""
    trace = []
    feedback = None

    for i in range(1, max_iters + 1):
        print(f"\n--- Iteration {i}/{max_iters} ---")

        # Generate
        draft = generate_draft(client, task, principles, feedback)
        print(f"Draft ({len(draft)} chars): {draft[:120]}...")

        # Verify
        result = call_verifier(draft, principles)
        verdict = result["verdict"]
        violations = verdict["violations"]
        confidence = verdict["confidence"]
        latency = result["latency_ms"]

        step = {
            "iteration": i,
            "draft": draft,
            "violations": violations,
            "confidence": confidence,
            "latency_ms": latency,
            "passed": len(violations) == 0,
        }
        trace.append(step)

        if not violations:
            print(f"PASSED at iteration {i} (confidence={confidence})")
            break
        else:
            print(f"FAILED: {violations} (confidence={confidence}, latency={latency}ms)")
            feedback = json.dumps(violations)
    else:
        print(f"\nMax iterations ({max_iters}) reached without passing.")

    return trace


def main():
    parser = argparse.ArgumentParser(description="Orchestrate draft refinement with VerifAI.")
    parser.add_argument("--task", required=True, help="The task / prompt for the frontier model.")
    parser.add_argument("--principles", required=True, help="Path to principles file.")
    parser.add_argument("--max_iters", type=int, default=3, help="Max refinement iterations.")
    parser.add_argument("--output", default="trace.jsonl", help="Output trace file (JSONL).")
    parser.add_argument("--print_trace", action="store_true", help="Print full trace to stdout.")
    args = parser.parse_args()

    principles_path = Path(args.principles)
    if not principles_path.exists():
        sys.exit(f"Error: principles file not found: {principles_path}")

    principles = [line.strip() for line in principles_path.read_text().splitlines() if line.strip()]
    if not principles:
        sys.exit("Error: principles file is empty.")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY environment variable is required.")

    client = OpenAI(api_key=api_key)

    trace = orchestrate(args.task, principles, args.max_iters, client)

    # Save trace
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for step in trace:
            f.write(json.dumps(step) + "\n")
    print(f"\nTrace saved to {output_path}")

    if args.print_trace:
        print("\n=== Full Trace ===")
        for step in trace:
            print(json.dumps(step, indent=2))

    # Exit code: 0 if passed, 1 if not
    if trace and trace[-1]["passed"]:
        print("\nFinal status: PASSED")
        sys.exit(0)
    else:
        print("\nFinal status: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
