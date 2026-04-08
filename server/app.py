# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Expense Audit Env Environment.

This module creates an HTTP server that exposes the ExpenseAuditEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - GET /tasks: List available tasks with grader info
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import ExpenseAuditAction, ExpenseAuditObservation
    from .environment import ExpenseAuditEnvironment
except ImportError:
    from models import ExpenseAuditAction, ExpenseAuditObservation
    from server.environment import ExpenseAuditEnvironment


# Create the app with web interface and README integration
app = create_app(
    ExpenseAuditEnvironment,
    ExpenseAuditAction,
    ExpenseAuditObservation,
    env_name="expense_audit_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


# ── Custom /tasks endpoint so the evaluator can discover graded tasks ──
@app.get("/tasks", tags=["Environment Info"])
def list_tasks():
    """List available tasks with grader information.
    
    The evaluator checks this endpoint to discover which tasks have graders.
    """
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Single clean report",
                "description": "One employee report with 2 perfect receipts. Approve after verification.",
                "difficulty": "easy",
                "has_grader": True,
            },
            {
                "id": "medium",
                "name": "Minor policy violations",
                "description": "Report with over-limit meal and one missing receipt. Partial approve + flag.",
                "difficulty": "medium",
                "has_grader": True,
            },
            {
                "id": "hard",
                "name": "Batch with fraud & duplicates",
                "description": "8 reports containing borderline amounts, duplicate receipts, missing receipts, and split-billing tricks.",
                "difficulty": "hard",
                "has_grader": True,
            },
        ]
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m expense_audit_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn expense_audit_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
