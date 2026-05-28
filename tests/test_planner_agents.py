from pathlib import Path

from assistant.agents import run_multi_agent_workflow
from assistant.memory import MemoryStore
from assistant.planner import TaskComplexity, advance_task, generate_plan, render_plan_debug


def test_planner_detects_simple_and_complex_tasks() -> None:
    assert generate_plan("Say hello").complexity == TaskComplexity.simple
    assert generate_plan("Build a multi-step plan for a complex project").complexity == TaskComplexity.complex


def test_planner_retry_limit_stops_failed_step() -> None:
    state = generate_plan("Say hello")

    state = advance_task(state, step_success=False, note="first failure")
    state = advance_task(state, step_success=False, note="second failure")

    assert state.failed
    assert not state.completed


def test_plan_debug_visualization() -> None:
    state = generate_plan("Build a multi-step plan for a complex project")
    debug = render_plan_debug(state)

    assert "Plan Debug" in debug
    assert "tools-disabled" in debug


def test_multi_agent_workflow_logs_messages(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "test.sqlite3")

    result = run_multi_agent_workflow("Plan a safe memory improvement", store)

    assert result["task_id"]
    assert "planner_agent" in result["agents"]
