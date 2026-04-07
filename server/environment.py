"""
Expense Audit Environment (FinOps-Audit-Env)
Real-world corporate expense report auditing.
"""
import random
from typing import Dict, Any

# NOTE: `from openenv import Environment` does NOT work (ImportError).
# The correct import path in openenv-core 0.2.3 is:
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import Action, Observation, Reward


class ExpenseAuditEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.seed = 42
        self.task_data = {
            "easy": self._generate_easy_data(),
            "medium": self._generate_medium_data(),
            "hard": self._generate_hard_data(),
        }
        self.current_task = None
        self._state = {}
        self._internal_state = State(episode_id="", step_count=0)

    def _generate_easy_data(self):
        return {
            "reports": [
                {"id": "R001", "employee": "Alice", "total": 120.50,
                 "items": [{"cat": "meals", "amt": 45.0, "rec": "REC1"},
                           {"cat": "travel", "amt": 75.5, "rec": "REC2"}],
                 "golden": "approve"}
            ],
            "policy": {"meals_max": 50, "travel_max": 100, "total_max": 200}
        }

    def _generate_medium_data(self):
        return {
            "reports": [
                {"id": "R002", "employee": "Bob", "total": 135.0,
                 "items": [{"cat": "meals", "amt": 68.0, "rec": None},
                           {"cat": "supplies", "amt": 67.0, "rec": "REC5"}],
                 "golden": "reject"}
            ],
            "policy": {"meals_max": 50, "travel_max": 100, "total_max": 200}
        }

    def _generate_hard_data(self):
        return {
            "reports": [
                # R003: APPROVE — looks suspicious (high total $195) but every item is within limits
                {"id": "R003", "employee": "Carol", "total": 195.0, "golden": "approve",
                 "items": [{"cat": "meals", "amt": 49.50, "rec": "REC3"}, {"cat": "travel", "amt": 99.50, "rec": "REC6"},
                           {"cat": "supplies", "amt": 46.0, "rec": "REC10"}]},
                # R004: REJECT — meals $51 just barely over $50 limit (borderline trap)
                {"id": "R004", "employee": "David", "total": 51.0, "golden": "reject",
                 "items": [{"cat": "meals", "amt": 51.0, "rec": "REC4"}]},
                # R005: REJECT — total $205 exceeds $200, but individual items are under limits
                {"id": "R005", "employee": "Eve", "total": 205.0, "golden": "reject",
                 "items": [{"cat": "travel", "amt": 95.0, "rec": "REC7"}, {"cat": "meals", "amt": 48.0, "rec": "REC11"},
                           {"cat": "supplies", "amt": 62.0, "rec": "REC8"}]},
                # R006: APPROVE — small clean report
                {"id": "R006", "employee": "Frank", "total": 40.0, "golden": "approve",
                 "items": [{"cat": "meals", "amt": 40.0, "rec": "REC9"}]},
                # R007: REJECT — duplicate receipt REC3 (same as Carol's R003 — fraud)
                {"id": "R007", "employee": "Grace", "total": 80.0, "golden": "reject",
                 "items": [{"cat": "meals", "amt": 30.0, "rec": "REC3"}, {"cat": "travel", "amt": 50.0, "rec": "REC12"}]},
                # R008: REJECT — missing receipt (rec=None)
                {"id": "R008", "employee": "Hank", "total": 75.0, "golden": "reject",
                 "items": [{"cat": "travel", "amt": 75.0, "rec": None}]},
                # R009: APPROVE — all items valid, slightly under every limit
                {"id": "R009", "employee": "Irene", "total": 148.0, "golden": "approve",
                 "items": [{"cat": "meals", "amt": 49.0, "rec": "REC13"}, {"cat": "travel", "amt": 99.0, "rec": "REC14"}]},
                # R010: REJECT — split-billing trick: two meals to avoid per-item limit
                # but total meals = $48+$47 = $95, which suggests gaming the system
                # Policy: per-item meals_max=50, but the employee is trying to split
                {"id": "R010", "employee": "Jake", "total": 95.0, "golden": "reject",
                 "items": [{"cat": "meals", "amt": 48.0, "rec": "REC15"}, {"cat": "meals", "amt": 47.0, "rec": "REC16"}]},
            ],
            "policy": {"meals_max": 50, "travel_max": 100, "total_max": 200}
        }

    def reset(self, task_id: str = "easy", **kwargs) -> Observation:
        random.seed(self.seed)
        self.current_task = task_id
        data = self.task_data[task_id]
        self._state = {
            "pending_reports": data["reports"],
            "processed": {},
            "decisions": {},
            "step_count": 0,
            "data": data
        }
        self._internal_state = State(episode_id=task_id, step_count=0)
        self._state["goal"] = f"Audit all expense reports for task '{task_id}'. Verify receipts, enforce policy, and make correct approve/reject decisions."

        return Observation(
            pending_reports=[{"id": r["id"], "total": r["total"]} for r in data["reports"]],
            policy_snapshot=data["policy"],
            goal=f"Audit all expense reports for task '{task_id}'. Verify receipts, enforce policy, and make correct approve/reject decisions.",
            last_feedback="Episode started.",
            done=False,
            reward=0.0,
        )

    # NOTE: Server-side step() must return Observation (not StepResult).
    # The HTTP server calls serialize_observation() on the return value,
    # which requires .model_dump(), .reward, .done — only Observation has these.
    # StepResult is a plain dataclass used CLIENT-SIDE only.
    def step(self, action: Action, **kwargs) -> Observation:
        self._state["step_count"] += 1
        self._internal_state.step_count += 1
        data = self._state["data"]
        report = next((r for r in data["reports"] if r["id"] == action.report_id), None)

        reward_value = 0.0
        info = {"correct_decision": False, "violation_detected": False}
        feedback = "Unknown action"

        if action.action_type == "view_report":
            if report is None:
                reward_value = -0.1
                feedback = f"Report {action.report_id} not found"
            else:
                reward_value = 0.15
                # Store report without golden answer for the observation
                safe_report = {k: v for k, v in report.items() if k != "golden"}
                self._state["current_report"] = safe_report
                feedback = f"Viewed report {action.report_id}"
        elif action.action_type == "check_policy":
            reward_value = 0.25
            feedback = "Policy checked"
        elif action.action_type in ["approve", "reject"]:
            correct = bool(report and report["golden"] == action.action_type)
            info["correct_decision"] = correct
            self._state["decisions"][action.report_id] = {"correct": correct, "golden": report["golden"] if report else None}
            reward_value = 0.45 if correct else -0.25
            decision_word = "Approved" if action.action_type == "approve" else "Rejected"
            feedback = f"{decision_word} report (correct: {correct})"
            # Mark as processed regardless of correctness — grader needs all reports decided
            self._state["processed"][action.report_id] = action.action_type
        elif action.action_type == "flag_duplicate":
            info["violation_detected"] = True
            reward_value = 0.35
            feedback = "Duplicate flagged"
        else:
            reward_value = 0.1
            feedback = f"Action {action.action_type} executed"

        # Anti-loop penalty
        if self._state["step_count"] > 30:
            reward_value -= 0.1

        done = len(self._state["processed"]) == len(data["reports"])

        # Deterministic grader (required for validator)
        if done:
            total = len(data["reports"])
            correct_count = sum(1 for d in self._state["decisions"].values() if d["correct"])
            steps = self._state["step_count"]

            # Accuracy component (0.0–1.0)
            accuracy = correct_count / total

            # Efficiency component — harder tasks get stricter step budgets
            difficulty_multiplier = {"easy": 10.0, "medium": 6.0, "hard": 4.0}
            budget = total * difficulty_multiplier.get(self.current_task, 6.0)
            efficiency = max(0.0, 1.0 - (steps / budget))

            # Final score: accuracy dominates, efficiency is a tiebreaker
            # Clamp to (0.01, 0.99) — validator requires strictly between 0 and 1
            grader_score = (0.7 * accuracy) + (0.3 * efficiency)
            grader_score = max(0.01, min(0.99, grader_score))

            info["grader_score"] = grader_score
            info["details"] = {
                "correct": correct_count, "total": total,
                "steps": steps, "accuracy": round(accuracy, 3),
                "efficiency": round(efficiency, 3),
            }

        return Observation(
            pending_reports=[
                {k: v for k, v in r.items() if k != "golden"}
                for r in data["reports"]
                if r["id"] not in self._state["processed"]
            ],
            current_report=self._state.get("current_report"),
            current_receipts=[],
            policy_snapshot=data["policy"],
            goal=self._state.get("goal", ""),
            last_feedback=feedback,
            done=done,
            reward=reward_value,
            metadata=info,
        )

    # NOTE: OpenEnv base class defines `state` as an abstract @property
    # returning State. A plain `def state()` would shadow the property
    # and break the framework.
    @property
    def state(self) -> State:
        return self._internal_state
