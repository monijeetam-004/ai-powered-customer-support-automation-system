"""
Shared pytest setup.

- Makes the project root importable (the modules live at the top level).
- Forces the offline, rule-based path by removing ANTHROPIC_API_KEY *before*
  llm.py is imported, so tests never call a paid API and stay deterministic.
- Runs every test inside its own temporary directory, so memory.db and the
  LangGraph checkpoint DB are created there and never touch your real files.
"""

import os
import sys
import uuid

import pytest

os.environ.pop("ANTHROPIC_API_KEY", None)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A freshly compiled graph whose databases live in a temp directory."""
    monkeypatch.chdir(tmp_path)  # memory.py uses the relative path "memory.db"

    import memory
    from graph import build_graph

    memory.init_db()
    return build_graph(checkpoint_path=str(tmp_path / "checkpoints.db"))


@pytest.fixture
def run_query(app):
    """Run one query; if the graph pauses for approval, resume it with `decision`.

    Pass decision=None to leave the request paused at the approval step.
    Returns the final graph state.
    """
    from langgraph.types import Command

    def _run(query, decision=None, customer_id="CUST-TEST", customer_name="Test"):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = {
            "customer_id": customer_id,
            "customer_name": customer_name,
            "query": query,
            "trace": [],
        }
        result = app.invoke(state, config=config)
        if "__interrupt__" in result and decision is not None:
            result = app.invoke(Command(resume=decision), config=config)
        return result

    return _run
