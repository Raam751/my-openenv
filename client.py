"""Expense Audit Environment Client."""

from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ExpenseAuditAction, ExpenseAuditObservation


class ExpenseAuditEnv(
    EnvClient[ExpenseAuditAction, ExpenseAuditObservation, State]
):
    """
    Client for the Expense Audit Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> async with ExpenseAuditEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset(task_id="easy")
        ...     print(result.observation.goal)
        ...
        ...     result = await client.step(ExpenseAuditAction(
        ...         report_id="R001", action_type="view_report"
        ...     ))
        ...     print(result.observation.last_feedback)
    """

    def _step_payload(self, action: ExpenseAuditAction) -> Dict[str, Any]:
        """Convert ExpenseAuditAction to JSON payload for step message."""
        payload: Dict[str, Any] = {
            "report_id": action.report_id,
            "action_type": action.action_type,
        }
        if action.fields is not None:
            payload["fields"] = action.fields
        if action.reason is not None:
            payload["reason"] = action.reason
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ExpenseAuditObservation]:
        """Parse server response into StepResult[ExpenseAuditObservation]."""
        obs_data = payload.get("observation", {})
        observation = ExpenseAuditObservation(
            pending_reports=obs_data.get("pending_reports", []),
            current_report=obs_data.get("current_report"),
            current_receipts=obs_data.get("current_receipts", []),
            policy_snapshot=obs_data.get("policy_snapshot", {}),
            goal=obs_data.get("goal", ""),
            last_feedback=obs_data.get("last_feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
