"""
Baseline inference script for the Expense Audit Environment.

Uses the OpenAI API client to run a model against all 3 tasks
and produce reproducible baseline scores.

MANDATORY env vars:
    API_BASE_URL  — The API endpoint for the LLM.
    MODEL_NAME    — The model identifier to use for inference.
    HF_TOKEN      — Your Hugging Face / API key.
    ENV_BASE_URL  — URL of the running environment server (e.g. your HF Space).
"""

import os   
import json
import asyncio
from openai import OpenAI
from models import Action
from client import ExpenseAuditEnv

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:8000"

MAX_STEPS = 15
TEMPERATURE = 0.2

SYSTEM_PROMPT = """You are an expert corporate expense auditor.
You will be given pending expense reports, policy rules, and receipts.
Your job is to view reports, check policy, flag issues, and approve or reject.

Reply with EXACTLY one valid JSON action object with these fields:
- "report_id": string (e.g. "R001")
- "action_type": one of "view_report", "view_receipt", "match_receipt", "check_policy", "approve", "reject", "flag_duplicate", "request_more_info"
- "fields": optional dict (e.g. {"receipt_id": "REC1"})
- "reason": optional string explaining your decision

Do not add any extra text or explanation. Only output the JSON object."""


def parse_action(text: str, fallback_report_id: str = "R001") -> Action:
    """Parse model response into an Action."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != 0:
            data = json.loads(text[start:end])
            return Action.model_validate(data)
    except Exception:
        pass
    return Action(
        report_id=fallback_report_id,
        action_type="view_report",
        reason="fallback",
    )


async def run_task(task_id: str) -> float:
    """Run a single task via the remote client and return grader score (0.0–1.0)."""
    async with ExpenseAuditEnv(base_url=ENV_BASE_URL) as env:
        result = await env.reset(task_id=task_id)
        obs = result.observation
        print(f"\n{'='*40}")
        print(f"  TASK: {task_id.upper()}")
        print(f"{'='*40}")
        print(f"Goal: {obs.goal}")

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        first_report_id = obs.pending_reports[0]["id"] if obs.pending_reports else "R001"
        grader_score = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                print("Environment signalled done.")
                break

            prompt = f"""Step: {step}
Goal: {obs.goal}
Pending reports: {json.dumps(obs.pending_reports)}
Current report: {json.dumps(obs.current_report) if obs.current_report else "None"}
Current receipts: {json.dumps(obs.current_receipts) if obs.current_receipts else "[]"}
Policy: {json.dumps(obs.policy_snapshot)}
Last feedback: {obs.last_feedback}

Reply with only a single JSON object."""

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=300,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                print(f"  Model request failed ({exc}). Using fallback.")
                response_text = ""

            action = parse_action(response_text, fallback_report_id=first_report_id)
            print(f"Step {step}: {action.action_type} on {action.report_id}")

            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            info = obs.metadata if hasattr(obs, "metadata") else {}

            print(f"  → Reward: {reward:+.2f} | Done: {result.done} | Info: {info}")

            # When done, the environment's grader score is in metadata
            if result.done:
                grader_score = info.get("grader_score", 0.0)
                print(f"Episode complete. Details: {info.get('details', {})}")
                break

        print(f"\nGRADER SCORE for {task_id}: {grader_score:.3f}")
        return grader_score


async def main():
    scores = {}
    for task in ["easy", "medium", "hard"]:
        scores[task] = await run_task(task)
    print(f"\n{'='*40}")
    print("  BASELINE SCORES")
    print(f"{'='*40}")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
