---
title: Expense Audit Env Environment Server
emoji: 💰
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Expense Audit Environment (FinOps-Audit-Env)

A real-world **corporate expense report auditing** environment for training and evaluating AI agents. Agents verify receipts, enforce spending policy, detect fraud/duplicates, and approve or reject employee expense reports — mirroring the exact daily workflow of finance teams worldwide.

## Motivation

Expense auditing is a massive real-world pain point. Companies like SAP Concur and Expensify have built multi-billion dollar businesses automating this workflow. This environment provides a standardized OpenEnv interface for training RL/LLM agents on the core audit task: given reports, receipts, and policy rules, make correct approve/reject decisions efficiently.

## Action Space

**`ExpenseAuditAction`** — what the agent can do each step:

| Field | Type | Description |
|---|---|---|
| `report_id` | `str` | Which report to act on (e.g. `"R001"`) |
| `action_type` | `Literal` | One of: `view_report`, `view_receipt`, `match_receipt`, `check_policy`, `approve`, `reject`, `flag_duplicate`, `request_more_info` |
| `fields` | `Optional[Dict]` | Extra parameters (e.g. `{"receipt_id": "REC1"}`) |
| `reason` | `Optional[str]` | Justification for the action |

## Observation Space

**`ExpenseAuditObservation`** — what the agent sees after each action:

| Field | Type | Description |
|---|---|---|
| `pending_reports` | `List[Dict]` | Summary of unprocessed reports (`id`, `employee`, `total`) |
| `current_report` | `Optional[Dict]` | Full detail of the currently viewed report |
| `current_receipts` | `List[Dict]` | Receipts for the current report |
| `policy_snapshot` | `Dict` | Active spending policy rules and limits |
| `goal` | `str` | Task objective description |
| `last_feedback` | `str` | Feedback from the previous action |
| `done` | `bool` | Whether the episode is complete |
| `reward` | `float` | Reward signal from the last action |

## Tasks

| Task ID | Name | Difficulty | Description |
|---|---|---|---|
| `easy` | Single clean report | Easy | One employee report with 2 perfect receipts. Verify and approve. |
| `medium` | Minor policy violations | Medium | Report with over-limit meal ($68 vs $50 limit) and a missing receipt. Agent must reject. |
| `hard` | Batch with fraud & duplicates | Hard | 4 reports containing duplicate receipts, over-limit spending, unverified receipts, and total exceeding policy. Correctly approve 2 and reject 2. |

## Reward Function

The reward function provides **partial progress signal** throughout the episode:

| Action | Reward | Notes |
|---|---|---|
| `view_report` | +0.15 | Encourages information gathering |
| `view_receipt` | +0.10 | Rewards due diligence |
| `match_receipt` | +0.20 | Receipt verification |
| `check_policy` | +0.25 | Policy compliance check |
| `approve` (correct) | **+0.45** | Correct decision |
| `reject` (correct) | **+0.45** | Correct decision |
| Wrong decision | **-0.25** | Penalty for incorrect approve/reject |
| `flag_duplicate` | +0.35 | Fraud detection |
| `request_more_info` | +0.05 | Minimal but non-zero |
| Anti-loop (>30 steps) | -0.10/step | Prevents infinite loops |

## Grader

Each task has a deterministic grader that scores agent performance from **0.0 to 1.0**:

```
score = (correct_decisions / total_reports) * 0.7 + efficiency_bonus * 0.3
```

- **Correct decisions**: approve/reject matching the ground truth
- **Efficiency bonus**: fewer steps = higher bonus (capped at 1.0)
- **Violations detected**: bonus for flagging duplicates

## Setup Instructions

### Prerequisites

- Python ≥ 3.10
- Docker (for containerized execution)

### Local Development

```bash
# Clone and enter the project
cd expense_audit_env

# Create virtual environment and install
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Run the server locally
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Validate OpenEnv compliance
openenv validate
```

### Docker

```bash
# Build
docker build -t expense-audit-env -f server/Dockerfile .

# Run
docker run --rm -p 8000:8000 expense-audit-env
```

### Running Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-hf-token"
export ENV_BASE_URL="http://localhost:8000"  # or your HF Space URL

python inference.py
```

### Deploy to Hugging Face Spaces

```bash
openenv push
```

## Baseline Scores

| Task | Expected Score | Notes |
|---|---|---|
| Easy | ~0.95 | Strong models approve after verification |
| Medium | ~0.80 | Requires detecting the policy violation |
| Hard | ~0.55 | Needs multi-report reasoning + duplicate detection |

## Project Structure

```
expense_audit_env/
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest with task definitions
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies
├── models.py              # Typed Action, Observation, Reward models
├── client.py              # ExpenseAuditEnv async WebSocket client
├── inference.py           # Baseline inference script (OpenAI API)
├── __init__.py            # Module exports
└── server/
    ├── __init__.py        # Server module exports
    ├── environment.py     # Core environment logic (step/reset/state)
    ├── app.py             # FastAPI application (HTTP + WebSocket)
    └── Dockerfile         # Container image definition
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset environment (pass `task_id` in body) |
| `/step` | POST | Execute an action |
| `/state` | GET | Get current environment state |
| `/health` | GET | Health check |
| `/schema` | GET | Action/observation JSON schemas |
| `/ws` | WS | WebSocket for persistent sessions |
| `/web` | GET | Interactive web UI |
