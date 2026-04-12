---
title: Email Triage Environment
emoji: 📧
colorFrom: red
colorTo: purple
sdk: docker
app_port: 7860
---

# Email Triage & Response Environment

An OpenEnv-compatible RL environment where an AI agent manages a realistic email inbox: reading messages, prioritising them, drafting replies, archiving junk, and flagging ambiguous items for human review.

Built for the **OpenEnv RL Challenge** hackathon.

---

## Motivation

Email triage is a real-world task that millions of knowledge workers do daily. It requires reading comprehension, priority assessment, professional writing, and judgment about what's spam vs. legitimate vs. ambiguous. This makes it an ideal testbed for evaluating LLM agent capabilities in a structured, scoreable way.

---

## Project Structure

```
email-triage-env/
├── inference.py       # LLM-powered agent (Groq via OpenAI client)
├── environment.py     # Core env: email data, action handling, graders
├── server.py          # FastAPI HTTP server (OpenEnv /reset, /step, /state, /score)
├── tests.py           # Unit test suite (python tests.py)
├── openenv.yaml       # OpenEnv task & resource manifest
├── .env               # API keys (not committed to git)
├── .gitignore
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## How It Works

The agent runs a standard RL loop against the environment:

```
                    ┌──────────────┐
                    │  LLM Agent   │
                    │ (inference)  │
                    └──────┬───────┘
                           │ JSON Action
                           ▼
                    ┌──────────────┐
                    │ Environment  │  ← reset() / step() / state() / score()
                    │ (email inbox)│
                    └──────┬───────┘
                           │ Observation + Reward
                           ▼
                    Back to Agent
```

1. `reset()` → loads the inbox, returns initial observation
2. Agent decides an action (list, read, label, reply, archive, flag)
3. `step(action)` → executes it, returns observation + reward
4. Repeat until the agent signals `done`
5. `score()` → returns final grade (0.0 – 1.0)

---

## Action Space

Every action is a JSON object with this schema:

```json
{
  "action": "<action_name>",
  "email_id": "<string or null>",
  "priority": "<urgent|normal|low or null>",
  "body": "<reply text or null>",
  "reason": "<flag reason or null>"
}
```

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `list_inbox` | — | List all emails with metadata (id, from, subject, labels) |
| `read` | `email_id` | Read the full body of a specific email |
| `label` | `email_id`, `priority` | Assign priority: `urgent`, `normal`, or `low` |
| `draft_reply` | `email_id`, `body` | Write and send a reply (must be >10 chars) |
| `archive` | `email_id` | Move email to archive (penalised if email is urgent) |
| `flag` | `email_id`, `reason` | Escalate for human review with a reason |

## Observation Space

Every step returns an observation with this schema:

```json
{
  "status": "ok | error | warning | done",
  "message": "Human-readable description of what happened",
  "data": { ... },
  "step_count": 5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `ok` (success), `error` (invalid action), `warning` (penalised action), `done` |
| `message` | string | Human-readable result of the action |
| `data` | dict or null | Structured data (email list, email body, label confirmation, etc.) |
| `step_count` | int | Current step number in the episode |

---

## Tasks

| # | Name | Difficulty | Emails | Max Steps | Description |
|---|------|-----------|--------|-----------|-------------|
| 1 | **Inbox Prioritisation** | Easy | 5 | 20 | Label each email as `urgent`, `normal`, or `low` |
| 2 | **Draft a Reply** | Medium | 1 | 10 | Reply to an angry customer complaint professionally |
| 3 | **Full Triage Pipeline** | Hard | 10 | 60 | Label all, reply to urgent, archive spam, flag ambiguous |

### Scoring (0.0 – 1.0)

```
Task 1 (Incremental):
  +0.2 per correct label (5 emails × 0.2 = max 1.0)

Task 2 (Checklist):
  +0.3  addresses all issues raised by customer
  +0.3  professional tone (formal language, empathy)
  +0.2  reply length & formatting (>50 chars)
  +0.2  no fabricated facts (no invented tracking numbers, dates, amounts)

Task 3 (Holistic):
  +0.50  correct priority labels (10 emails, normalised)
  +0.40  replies drafted for urgent emails (4 urgent emails)
  +0.10  archive spam + flag ambiguous
  -0.10  penalty per destructive action (e.g. archiving an urgent email)
  -0.05  penalty per looping/repeated action
```

All graders are **deterministic** — same actions always produce the same score.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```env
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
HF_TOKEN=your_groq_api_key_here
```

Get a free Groq API key at: [console.groq.com/keys](https://console.groq.com/keys)

### 3. Run the agent

```bash
# Set your API key (Linux/Mac)
export HF_TOKEN=gsk_your_key_here

# Set your API key (Windows PowerShell)
$env:HF_TOKEN="gsk_your_key_here"

# Run individual tasks
python inference.py --task 1    # easy
python inference.py --task 2    # medium
python inference.py --task 3    # hard

# Run all tasks and get aggregate scores
python inference.py --all
```

### 4. Run the tests

```bash
python tests.py
# Expected: 17/17 tests passed
```

### 5. Run the HTTP server

```bash
python server.py
# Listens on http://localhost:8000
```

Interact via HTTP:

```bash
# Reset task 1
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" \
     -d '{"task": 1}'

# Take a step
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
     -d '{"task": 1, "action": {"action": "list_inbox"}}'

# Get current score
curl http://localhost:8000/score?task=1
```

### 6. Docker

```bash
docker build -t email-triage-env .
docker run -p 8000:8000 email-triage-env
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | **Yes** | — | API key for the LLM provider (Groq key) |
| `API_BASE_URL` | No | `https://api.groq.com/openai/v1` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | No | `llama-3.3-70b-versatile` | Model to use for inference |

The hackathon runner injects `HF_TOKEN` automatically. `API_BASE_URL` and `MODEL_NAME` have sensible defaults.

---

## Baseline Scores

Scores from the baseline `inference.py` agent using **Llama 3.3 70B** on Groq:

| Task | Score | Steps Used | Notes |
|------|-------|------------|-------|
| 1 — Inbox Prioritisation | **1.00** | ~11 | All 5 labels correct |
| 2 — Draft a Reply | **0.90** | ~4 | Professional, addresses all issues |
| 3 — Full Triage Pipeline | **0.85** | ~35 | Labels + replies + archive + flag |

> These are representative scores. Actual scores may vary slightly due to LLM non-determinism at temperature 0.2.

---

## How This Would Work With Real Emails

This project is currently a **simulation** — the emails are hardcoded sample data inside `environment.py`. But the architecture is designed so it can be connected to a real email inbox with minimal changes.

### Connecting to a Real Email Provider

| Method | Best For | How |
|--------|----------|-----|
| **Gmail API** | Gmail / Google Workspace | `google-api-python-client` + OAuth2 |
| **Microsoft Graph API** | Outlook / Office 365 | REST API + app registration |
| **IMAP/SMTP** | Any provider | Python's built-in `imaplib` + `smtplib` |

### What Would Change

| Layer | Current (Hackathon) | Real-Life Version |
|-------|-------------------|------------------|
| **Email source** | Hardcoded Python dicts | Gmail API / IMAP / Outlook API |
| **Actions** | Modify in-memory objects | Call real email APIs (label, send, archive) |
| **AI brain** | Groq LLM | Same — no change needed |
| **Trigger** | Manual CLI command | Cron job, webhook, or always-on service |
| **Safety** | None needed (simulation) | Drafts-only mode, audit logs, undo window |

The **agent logic (`inference.py`) stays exactly the same** — only the environment layer needs to be swapped from simulated emails to real API calls.

### Example: Automated Morning Triage

```
You receive 50 emails overnight.

The agent runs automatically at 7 AM:
  ├── 8 marked "urgent"   → drafts ready for your review
  ├── 12 newsletters      → archived automatically
  ├── 3 suspicious emails → flagged for you to check
  ├── 25 normal emails    → labelled and sorted
  └── 2 ambiguous emails  → flagged with explanation

You wake up to 13 items needing attention instead of 50.
```

### Safety Guardrails for Production

- **Draft mode**: Save replies as drafts instead of auto-sending
- **Allowlist/blocklist**: Only act on specific senders/domains
- **Audit log**: Record every agent action for review
- **Undo window**: 60-second delay before sending
- **Cost monitoring**: Track API usage for free-tier limits

---

## Technical Notes

- **LLM Client**: `openai` Python SDK pointed at Groq's OpenAI-compatible endpoint
- **Model**: Llama 3.3 70B Versatile (hosted on Groq, free tier)
- **Retry Logic**: Exponential backoff (5s → 10s → 20s) on rate-limit errors
- **Pure Python**: No GPU required
- **Resources**: Runs within 2 vCPU / 4 GB RAM
- **Deterministic graders**: Same actions always produce the same score
- **Pydantic v2**: Typed models for Action, Observation, StepResult, InboxState
- **17 unit tests**: Full coverage of environment logic across all 3 tasks
