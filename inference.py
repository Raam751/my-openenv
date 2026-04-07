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

from server.environment import ExpenseAuditEnvironment
from models import Action

# --- Try to import the async client for Docker mode ---
try:
    from client import ExpenseAuditEnv

    _HAS_CLIENT = True
except Exception:
    _HAS_CLIENT = False

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # If you are using docker image
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME")
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

def build_user_prompt(step: int, obs_data: dict) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}
        Goal: {obs_data['goal']}
        Pending reports: {json.dumps(obs_data['pending_reports'])}
        Current report: {json.dumps(obs_data['current_report']) if obs_data['current_report'] else "None"}
        Current receipts: {json.dumps(obs_data['current_receipts']) if obs_data['current_receipts'] else "[]"}
        Policy: {json.dumps(obs_data['policy_snapshot'])}
        Last feedback: {obs_data['last_feedback']}

        Reply with only a single JSON object.
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, obs_data: dict, fallback_id: str) -> Action:
    """Call the LLM and parse the response into an Action."""
    user_prompt = build_user_prompt(step, obs_data)
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


# ───────────────────── Observation helper ─────────────────────

def obs_to_dict(obs):
    """Convert an Observation (Pydantic model or StepResult.observation) to a plain dict."""
    return {
        "goal": getattr(obs, "goal", ""),
        "pending_reports": getattr(obs, "pending_reports", []),
        "current_report": getattr(obs, "current_report", None),
        "current_receipts": getattr(obs, "current_receipts", []),
        "policy_snapshot": getattr(obs, "policy_snapshot", {}),
        "last_feedback": getattr(obs, "last_feedback", ""),
        "done": getattr(obs, "done", False),
        "reward": getattr(obs, "reward", 0.0),
        "metadata": getattr(obs, "metadata", {}),
    }


# ───────────────────── Direct mode (no server needed) ─────────────────────

def run_task_direct(task_id: str) -> float:
    """Run a single task using the environment directly (no Docker/server)."""
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = ExpenseAuditEnvironment()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id=task_id)
        obs_data = obs_to_dict(obs)

        first_report_id = obs_data["pending_reports"][0]["id"] if obs_data["pending_reports"] else "R001"

        for step in range(1, MAX_STEPS + 1):
            if obs_data["done"]:
                break

            action = get_model_action(llm, step, obs_data, fallback_id=first_report_id)
            action_str = f"{action.action_type}({action.report_id})"

            obs = env.step(action)
            obs_data = obs_to_dict(obs)

            reward = obs_data["reward"] or 0.0
            done = obs_data["done"]
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                info = obs_data["metadata"]
                score = info.get("grader_score", 0.01)
                break

        # If episode didn't complete, compute partial score from rewards
        if not done and rewards:
            # Normalize: positive rewards indicate progress, scale to (0, 1)
            total_reward = sum(rewards)
            max_possible = steps_taken * 0.45  # max reward per step
            score = max(0.01, min(0.50, total_reward / max_possible)) if max_possible > 0 else 0.01

        score = min(max(score, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ───────────────────── Docker mode (from_docker_image) ─────────────────────

async def run_task_docker(task_id: str) -> float:
    """Run a single task via Docker image (auto-starts container)."""
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await ExpenseAuditEnv.from_docker_image(LOCAL_IMAGE_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation
        obs_data = obs_to_dict(obs)

        first_report_id = obs_data["pending_reports"][0]["id"] if obs_data["pending_reports"] else "R001"

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(llm, step, obs_data, fallback_id=first_report_id)
            action_str = f"{action.action_type}({action.report_id})"

            result = await env.step(action)
            obs = result.observation
            obs_data = obs_to_dict(obs)

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                info = obs_data["metadata"]
                score = info.get("grader_score", 0.01)
                break

        # If episode didn't complete, compute partial score from rewards
        if not done and rewards:
            total_reward = sum(rewards)
            max_possible = steps_taken * 0.45
            score = max(0.01, min(0.50, total_reward / max_possible)) if max_possible > 0 else 0.01

        score = min(max(score, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ───────────────────── Entry point ─────────────────────

def main() -> None:
    # OpenEnv evaluator runs one task per inference script execution
    # Run a single task and return a single final score.
    task_id = os.getenv("TASK_ID", "easy")
    score = 0.0

    if LOCAL_IMAGE_NAME and _HAS_CLIENT:
        # Docker mode: auto-start container via from_docker_image()
        async def _run():
            return await run_task_docker(task_id)
        score = asyncio.run(_run())
    else:
        # Direct mode: use environment in-process (no server needed)
        score = run_task_direct(task_id)

    print(f"\nFinal score for {task_id}: {score:.3f}", flush=True)


if __name__ == "__main__":
    main()
