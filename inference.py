"""
inference.py -- Agent that solves all three Email Triage tasks.

Uses the OpenAI Python client pointed at a Groq-compatible endpoint.
All LLM config is controlled via environment variables:
    - API_BASE_URL : base URL for the OpenAI-compatible API  (has default)
    - MODEL_NAME   : model to use for inference               (has default)
    - HF_TOKEN     : API key (mandatory, injected by runner)

Usage:
    python inference.py --task 1   # run task 1
    python inference.py --task 2   # run task 2
    python inference.py --task 3   # run task 3 (full pipeline)
    python inference.py --all      # run all tasks and report scores
"""

import argparse
import json
import os
import time
from typing import Any

from openai import OpenAI

from environment import EmailTriageEnv, Action

# ---------------------------------------------------------------------------
# Configuration via environment variables (hackathon-compliant)
# ---------------------------------------------------------------------------

# API_BASE_URL: Groq's OpenAI-compatible endpoint (default provided)
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")

# MODEL_NAME: which model to use on the endpoint (default provided)
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")

# HF_TOKEN: the API key -- mandatory, injected by hackathon runner
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Initialize the OpenAI client pointing at Groq (or whatever API_BASE_URL is)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert email triage agent. You manage an inbox
efficiently by reading emails, assigning priority labels, drafting professional
replies, archiving junk, and flagging ambiguous items for human review.

You interact with an email environment through a strict JSON action interface.
Each response you produce MUST be a single valid JSON object -- no markdown,
no extra text -- in exactly this format:

{
  "action": "<action_name>",
  "email_id": "<id or null>",
  "priority": "<urgent|normal|low or null>",
  "body": "<reply text or null>",
  "reason": "<flag reason or null>"
}

Available actions:
- list_inbox   -- see all emails (no email_id needed)
- read         -- read full body of an email (requires email_id)
- label        -- assign a priority label (requires email_id + priority)
- draft_reply  -- write a reply (requires email_id + body)
- archive      -- move to archive (requires email_id)
- flag         -- escalate for human review (requires email_id + reason)

Rules:
- NEVER archive an urgent email.
- ALWAYS read an email before labelling or replying.
- Draft replies ONLY for urgent emails (unless instructed otherwise).
- Archive obvious spam/junk.
- Flag emails that are ambiguous or need human judgment.
- When drafting replies: be professional, address all issues raised, do NOT
  invent facts (no fake tracking numbers, refund amounts, dates).
- Signal completion by returning: {"action": "done", "email_id": null, "priority": null, "body": null, "reason": null}
"""

# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def parse_action(text: str) -> dict[str, Any]:
    """Extract JSON from model output (handles minor formatting noise)."""
    text = text.strip()
    # Strip markdown fences if present (some models wrap JSON in ```)
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```"))
    return json.loads(text)


def run_task(task: int, max_steps: int = 40, verbose: bool = True) -> float:
    """Run a single task with the LLM agent. Returns final score."""

    env = EmailTriageEnv(task=task)
    obs = env.reset()

    # --- Hackathon output marker ---
    print("[START]")

    if verbose:
        print(f"Task {task} | Model: {MODEL_NAME} | Endpoint: {API_BASE_URL}")

    task_instruction = {
        1: (
            "Task: Read all 5 emails and label each as urgent, normal, or low priority. "
            "Start by listing the inbox, then read each email before labelling it. "
            "When done, output the done action."
        ),
        2: (
            "Task: Read the customer complaint email and draft a professional reply "
            "that addresses ALL issues the customer raised. Be empathetic, professional, "
            "and do not invent any facts. When done, output the done action."
        ),
        3: (
            "Task: Full triage pipeline on 10 emails.\n"
            "1. List the inbox.\n"
            "2. Read each email.\n"
            "3. Label all emails (urgent / normal / low).\n"
            "4. Draft replies for urgent emails.\n"
            "5. Archive obvious spam / junk.\n"
            "6. Flag ambiguous emails for human review.\n"
            "When everything is done, output the done action."
        ),
    }[task]

    # Build the message history (OpenAI format: system + user/assistant turns)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_instruction},
    ]
    step = 0

    while step < max_steps:
        # Call the LLM via the OpenAI client (works with Groq, vLLM, etc.)
        # Retry with backoff on rate-limit errors (Groq free tier: 30 RPM)
        raw = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.2,
                )
                raw = response.choices[0].message.content
                break
            except Exception as e:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s
                if verbose:
                    print(f"  [RETRY] {type(e).__name__} -- waiting {wait}s (attempt {attempt+1}/3)")
                time.sleep(wait)

        if raw is None:
            if verbose:
                print("  [ERROR] LLM call failed after 3 retries. Ending task.")
            break

        messages.append({"role": "assistant", "content": raw})

        if verbose:
            print(f"[Step {step+1}] Agent: {raw[:120]}{'...' if len(raw) > 120 else ''}")

        # Parse action from model output
        try:
            action_dict = parse_action(raw)
        except json.JSONDecodeError as e:
            if verbose:
                print(f"  [WARN] JSON parse error: {e} -- asking agent to retry")
            messages.append({
                "role": "user",
                "content": f"Your last response was not valid JSON. Error: {e}. Please try again with a valid JSON action."
            })
            continue

        # Done?
        if action_dict.get("action") == "done":
            if verbose:
                print("  Agent signalled completion.")
            break

        # Execute action in the environment
        try:
            action = Action(**action_dict)
        except Exception as e:
            messages.append({"role": "user", "content": f"Invalid action format: {e}. Try again."})
            continue

        result = env.step(action)

        # --- Hackathon output marker ---
        print("[STEP]")

        if verbose:
            print(f"  Env: [{result.observation.status}] {result.observation.message}  reward={result.reward:+.2f}")

        # Feed observation back to the agent
        obs_summary = {
            "status": result.observation.status,
            "message": result.observation.message,
            "data": result.observation.data,
            "step": result.observation.step_count,
            "running_score": env.score(),
        }
        messages.append({"role": "user", "content": json.dumps(obs_summary)})
        step += 1

    final_score = env.score()

    # --- Hackathon output marker ---
    print("[END]")

    if verbose:
        print(f"Final score: {final_score:.2f} / 1.00  (steps used: {step})")

    return final_score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Email Triage Agent")
    parser.add_argument("--task", type=int, choices=[1, 2, 3],
                        help="Run a specific task (1, 2, or 3)")
    parser.add_argument("--all", action="store_true",
                        help="Run all three tasks")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.all:
        scores = {}
        for t in [1, 2, 3]:
            scores[t] = run_task(t, verbose=verbose)
        print("=" * 40)
        print("  FINAL SCORES")
        print("=" * 40)
        for t, s in scores.items():
            print(f"  Task {t}: {s:.2f}")
        print(f"  Average: {sum(scores.values()) / 3:.2f}")
    elif args.task:
        run_task(args.task, verbose=verbose)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
