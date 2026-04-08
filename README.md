---
title: ExpenseGuard Env
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

# ExpenseGuard Env

A real-world **corporate expense report auditing** environment for training and evaluating AI agents. Agents verify receipts, enforce strict spending policy limits, detect duplicate-receipt fraud, and make terminal `approve` or `reject` decisions — mirroring the exact daily workflow of finance teams worldwide.

## Motivation
Expense auditing is a massive real-world pain point, automated by companies like SAP Concur and Expensify. This environment provides a standardized OpenEnv interface for benchmarking Frontier LLM / RL Agents computationally. Given batches of JSON reports, categorical receipts, and dynamic policy rules, agents must explore the data structure and enforce exact logical compliance safely without hallucinating.

---

## Action Space

The action space is highly streamlined to eliminate agent hallucination and prevent zero-value loops.

| Action Type | Description |
|---|---|
| `view_report` | Fetches the full JSON details of a report ID. **(Mandatory before making a decision)** |
| `verify_receipts` | Cross-references attached receipts with items to ensure unique, valid matching. |
| `approve` | Terminal decision: The report is clean and explicitly complies with all policy rules. |
| `reject` | Terminal decision: The report contains a policy violation, missing evidence, or fraud. |

*Note: All actions optionally accept a `reason: str` field to log internal chain-of-thought.*

---

## Observation Space

What the agent sees after each action:

```json
{
  "pending_reports": [
    {"id": "R001", "total": 120.50},
    {"id": "R002", "total": 45.00}
  ],
  "current_report": {"id": "R001", "items": [{"cat": "meals", "amt": 45.0, "rec": "REC1"}]},
  "current_receipts": [{"id": "REC1", "status": "verified"}],
  "policy_snapshot": {
    "limits": {"meals_max": 50, "travel_max": 100, "total_max": 200},
    "rules": [
      "1. The sum of items in a category cannot exceed that category's limit.",
      "2. Every item must have a valid attached receipt.",
      "3. Submitting the same receipt ID across different reports is duplicate fraud."
    ]
  },
  "last_feedback": "Viewed report R001 successfully.",
  "goal": "Audit all expense reports for task 'easy'.",
  "done": false
}
```
*Note: To prevent reward hacking, the ground-truth golden label and decision accuracy flags are strictly scrubbed from the text observation!*

---

## Tasks

| Task ID | Difficulty | Description |
|---|---|---|
| `easy` | Easy | Single clean report with exactly 2 perfect receipts. Verify and approve. |
| `medium` | Medium | Report with an over-limit meal ($68 vs $50 limit) and a missing receipt. Agent must reject. |
| `hard` | Hard | A batch of 8 reports testing boundaries ($49.99 vs $50), explicit duplicate receipt fraud (`REC3` reused), and split-billing evasion tactics. |

---

## Scoring Logic & Grader

The deterministic grader calculates the score strictly from `0.00` to `0.99`, outputting explicitly trackable data-science metrics.

### 1. Correctness (90% Base Weight)
The core component is pure decision accuracy `(True Positives + True Negatives) / Total Reports`. 

### 2. Efficiency Tiebreaker (10% Base Weight)
Fast execution (fewer steps) provides a bonus. **Reward Farming Protection:** Efficiency points are aggressively withheld (`multiplied to 0.0`) if the agent's baseline accuracy drops below `0.75`. Blind-guess scripts receive exactly `0.0` bonus.

### 3. Evidentiary Penalties
If an agent executes `approve` or `reject` without first invoking `view_report`, it is immediately graded as a `-1.0` blind-guess failure. The agent must physically observe evidence.

### 4. Explicit Fraud Bonus (Hard Task Only)
Because `hard` contains specific nuanced traps, `20%` of the grading weight is converted into an explicit boolean bonus for safely rejecting the Duplicate Fraud report (`R007`) and the Split-Billing Evasion report (`R010`).

---

## Setup & Testing Instructions

### Prerequisites
- Python ≥ 3.10
- `uv` package manager (recommended) or pip

### Run Locally (Development)
```bash
# Enter project root
cd expense_audit_env

# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -e .

# Start the environment server directly via FastAPI
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### How to Test (Baseline Inference)
The repository includes an `inference.py` script that acts as the evaluator wrapper. You can test your tasks natively without connecting to OpenEnv's remote orchestrator.

```bash
# Set your model provider
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model"
export HF_TOKEN="your-token"

# Run a specific task natively locally
python inference.py --task hard
```

### How to Deploy
Ensure you are authenticated with the Hugging Face CLI, then push the OpenEnv template seamlessly:
```bash
openenv build
openenv validate
openenv push
```

---

## Known Limitations
- **Stateless WebSockets:** Because the environment runs across HTTP validation endpoints, `step()` gracefully enforces its own auto-reset wrapper if a user bypasses `reset()`, but extreme parallel bursts of HTTP calls without `episode_id` scoping could theoretically misalign `self._state`.
- **Text Reasoning vs API Formatting:** Currently, Models strictly output `Action` JSON payloads. If they lack strong strict-JSON compliance tools, they may struggle to execute the schema without formatting wrappers. 
- **Receipt Parsing:** The receipts are currently simulated dictionaries. A future iteration of this project would ideally require OCR analysis on actual PNG receipts.

---
*Developed for the OpenEnv Multi-Modal Agentic Benchmark.*
