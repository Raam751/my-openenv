"""
Data models for the Expense Audit Environment.

Typed Pydantic models for action, observation, and reward
used by the FinOps expense-report auditing environment.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal

from openenv.core.env_server.types import Action as _BaseAction, Observation as _BaseObservation


class Action(_BaseAction):
    """An auditor action on an expense report."""

    report_id: str
    action_type: Literal[
        "view_report", 
        "view_receipt",
        "verify_receipts", 
        "check_policy",
        "flag_duplicate",
        "request_more_info",
        "approve", 
        "reject"
    ]
    fields: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


class Observation(_BaseObservation):
    """What the auditor agent sees after each action."""

    pending_reports: List[Dict[str, Any]] = []
    current_report: Optional[Dict[str, Any]] = None
    current_receipts: List[Dict[str, Any]] = []
    policy_snapshot: Dict[str, Any] = {}
    goal: str = ""
    last_feedback: str = ""
    grader_score: Optional[float] = None  # Exposed as a top-level field so it survives serialization


class Reward(BaseModel):
    """Shaped reward with transparency info dict."""

    value: float
    info: Dict[str, Any] = {}


# Aliases used by app.py and client.py
ExpenseAuditAction = Action
ExpenseAuditObservation = Observation
