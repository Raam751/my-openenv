"""
Expense Audit Environment (FinOps-Audit-Env)
Real-world corporate expense report auditing.
"""
import random
from typing import Dict, Any, Optional

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
            "policy": {
                "limits": {"meals_max": 50, "travel_max": 100, "total_max": 200},
                "rules": [
                    "1. The sum of items in a category cannot exceed that category's limit.",
                    "2. The total report sum cannot exceed total_max.",
                    "3. Every item must have a valid attached receipt.",
                    "4. Submitting the same receipt ID across different reports is duplicate fraud and must be rejected.",
                    "5. Split-billing (submitting multiple small items in the same category) is only valid if their SUM is under the category limit."
                ]
            }
        }

    def _generate_medium_data(self):
        return {
            "reports": [
                {"id": "R002", "employee": "Bob", "total": 135.0,
                 "items": [{"cat": "meals", "amt": 68.0, "rec": None},
                           {"cat": "supplies", "amt": 67.0, "rec": "REC5"}],
                 "golden": "reject"}
            ],
            "policy": {
                "limits": {"meals_max": 50, "travel_max": 100, "total_max": 200},
                "rules": [
                    "1. The sum of items in a category cannot exceed that category's limit.",
                    "2. The total report sum cannot exceed total_max.",
                    "3. Every item must have a valid attached receipt.",
                    "4. Submitting the same receipt ID across different reports is duplicate fraud and must be rejected.",
                    "5. Split-billing (submitting multiple small items in the same category) is only valid if their SUM is under the category limit."
                ]
            }
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
            "policy": {
                "limits": {"meals_max": 50, "travel_max": 100, "total_max": 200},
                "rules": [
                    "1. The sum of items in a category cannot exceed that category's limit.",
                    "2. The total report sum cannot exceed total_max.",
                    "3. Every item must have a valid attached receipt.",
                    "4. Submitting the same receipt ID across different reports is duplicate fraud and must be rejected.",
                    "5. Split-billing (submitting multiple small items in the same category) is only valid if their SUM is under the category limit."
                ]
            }
        }

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, task_id: Optional[str] = None, **kwargs) -> Observation:
        task_id = episode_id or task_id or "easy"
        if task_id not in self.task_data:
            task_id = "easy"
            
        if seed is not None:
            self.seed = seed
            
        random.seed(self.seed)
        self.current_task = task_id
        data = self.task_data[task_id]
        self._state = {
            "pending_reports": data["reports"],
            "processed": {},
            "decisions": {},
            "step_count": 0,
            "reports_viewed": set(),
            "receipts_verified": set(),
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
        # Auto-reset if step is called without a prior reset (stateless HTTP mode)
        if not self._state:
            self.reset(task_id="easy")
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
                if action.report_id in self._state["reports_viewed"]:
                    reward_value = -0.1  # Penalize repetition
                else:
                    reward_value = 0.05  # Small reward for initial exploration
                
                safe_report = {k: v for k, v in report.items() if k != "golden"}
                self._state["current_report"] = safe_report
                self._state["reports_viewed"].add(action.report_id)
                feedback = f"Viewed report {action.report_id}"
        elif action.action_type == "verify_receipts":
            if report is None:
                feedback = f"Report {action.report_id} not found."
                reward_value = -0.1
            else:
                if action.report_id in self._state["receipts_verified"]:
                    reward_value = -0.1  # Penalize repetition
                else:
                    reward_value = 0.05  # Small initial reward
                    
                self._state["reports_viewed"].add(action.report_id)
                self._state["receipts_verified"].add(action.report_id)
                receipt_ids = [item.get("rec") for item in report.get("items", []) if item.get("rec")]
                
                if not receipt_ids:
                    feedback = f"Failure: No valid receipts attached to {action.report_id}."
                else:
                    self._state["current_receipts"] = [{"id": rid, "status": "verified"} for rid in receipt_ids]
                    feedback = f"Success: Verified {len(receipt_ids)} unique receipts for {action.report_id}."
        elif action.action_type in ["approve", "reject"]:
            viewed_first = action.report_id in self._state["reports_viewed"]
            is_correct_match = bool(report and report["golden"] == action.action_type)
            correct = viewed_first and is_correct_match

            info["correct_decision"] = correct
            self._state["decisions"][action.report_id] = {
                "correct": correct, 
                "golden": report["golden"] if report else None,
                "viewed_first": viewed_first
            }
            
            if not viewed_first:
                reward_value = -1.0 # Heavy penalty for blind guessing
                feedback = f"System Error: Cannot process decision. Report {action.report_id} must be viewed first."
            else:
                # Terminal decision matters most
                reward_value = 1.0 if correct else -1.0
                decision_word = "Approved" if action.action_type == "approve" else "Rejected"
                feedback = f"{decision_word} report {action.report_id} successfully."
                
            # Mark as processed regardless of correctness — grader needs all reports decided
            self._state["processed"][action.report_id] = action.action_type
        else:
            reward_value = -0.1
            feedback = f"Invalid action: {action.action_type}"

        # Anti-loop penalty
        if self._state["step_count"] > 30:
            reward_value -= 0.1

        done = len(self._state["processed"]) == len(data["reports"])

        # Deterministic grader (required for validator)
        if done:
            total = len(data["reports"])
            decisions = self._state["decisions"]
            correct_count = sum(1 for d in decisions.values() if d["correct"])
            steps = self._state["step_count"]

            # Accuracy strongly dominates
            accuracy = correct_count / total

            # Efficiency acts purely as a tiebreaker/penalty (never outweighs accuracy)
            budget = total * 10.0  # Generous budget to encourage thorough checking
            efficiency = max(0.0, 1.0 - (steps / budget))

            # Base Weights and Formula
            accuracy_weight = 0.90
            efficiency_weight = 0.10
            fraud_bonus = 0.0

            # Explicit bonus for fraud/duplicate detection on the Hard task
            if self.current_task == "hard":
                accuracy_weight = 0.70  # Shift 20% weight to strict fraud/duplicate detection bonuses
                caught_dup = decisions.get("R007", {}).get("correct", False)
                caught_split = decisions.get("R010", {}).get("correct", False)
                
                if caught_dup: fraud_bonus += 0.10
                if caught_split: fraud_bonus += 0.10

            # Prevent fast-guessing reward hacking: Efficiency is ONLY rewarded if accuracy is very high
            actual_efficiency = efficiency * efficiency_weight if accuracy >= 0.75 else 0.0

            grader_score = (accuracy_weight * accuracy) + actual_efficiency + fraud_bonus
            grader_score = max(0.01, min(0.99, grader_score))

            info["grader_score"] = grader_score
            info["details"] = {
                "correct": correct_count, 
                "total": total,
                "steps": steps, 
                "accuracy": round(accuracy, 3),
                "efficiency_bonus": round(actual_efficiency, 3),
                "fraud_bonus": round(fraud_bonus, 3)
            }

        # When episode is done, use grader_score as the reward so evaluator can read it
        final_grader = info.get("grader_score", None)
        if done and final_grader is not None:
            reward_value = final_grader

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
            grader_score=final_grader,
        )

    # NOTE: OpenEnv base class defines `state` as an abstract @property
    # returning State. A plain `def state()` would shadow the property
    # and break the framework.
    @property
    def state(self) -> State:
        return self._internal_state
