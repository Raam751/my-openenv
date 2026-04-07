"""
Inference Script — Expense Audit Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]
"""

import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import ExpenseAuditEnv
from models import Action

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # If you are using docker image
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("EXPENSE_AUDIT_BENCHMARK", "expense_audit_env")
MAX_STEPS = 15
TEMPERATURE = 0.2
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert corporate expense auditor.
    You will be given pending expense reports, policy rules, and receipts.
    Your job is to view reports, check policy, flag issues, and approve or reject.

    Reply with EXACTLY one valid JSON action object with these fields:
    - "report_id": string (e.g. "R001")
    - "action_type": one of "view_report", "view_receipt", "match_receipt",
      "check_policy", "approve", "reject", "flag_duplicate", "request_more_info"
    - "fields": optional dict (e.g. {"receipt_id": "REC1"})
    - "reason": optional string explaining your decision

    Do not add any extra text or explanation. Only output the JSON object.
    """
).strip()


# ───────────────────── Logging helpers (mandatory stdout format) ─────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ───────────────────── Action parsing ─────────────────────

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


# ───────────────────── Prompt building ─────────────────────

def build_user_prompt(step: int, obs) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}
        Goal: {obs.goal}
        Pending reports: {json.dumps(obs.pending_reports)}
        Current report: {json.dumps(obs.current_report) if obs.current_report else "None"}
        Current receipts: {json.dumps(obs.current_receipts) if obs.current_receipts else "[]"}
        Policy: {json.dumps(obs.policy_snapshot)}
        Last feedback: {obs.last_feedback}

        Reply with only a single JSON object.
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, obs, fallback_id: str) -> Action:
    """Call the LLM and parse the response into an Action."""
    user_prompt = build_user_prompt(step, obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_action(text, fallback_report_id=fallback_id)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return Action(report_id=fallback_id, action_type="view_report", reason="fallback")


# ───────────────────── Main loop ─────────────────────

async def run_task(task_id: str) -> float:
    """Run a single task episode and return a score in [0, 1]."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if LOCAL_IMAGE_NAME:
        env = await ExpenseAuditEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = ExpenseAuditEnv(base_url=os.getenv("ENV_BASE_URL", "http://localhost:8000"))

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        first_report_id = obs.pending_reports[0]["id"] if obs.pending_reports else "R001"

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(client, step, obs, fallback_id=first_report_id)
            action_str = f"{action.action_type}({action.report_id})"

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                # Get grader score from environment metadata
                info = obs.metadata if hasattr(obs, "metadata") else {}
                score = info.get("grader_score", 0.0)
                break

        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    scores = {}
    for task in ["easy", "medium", "hard"]:
        scores[task] = await run_task(task)
    print(f"\nFinal scores: {json.dumps(scores, indent=2)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
