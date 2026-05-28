"""Hierarchical task planning without autonomous tool execution."""

from __future__ import annotations

from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskComplexity(str, Enum):
    """Task complexity labels."""

    simple = "simple"
    complex = "complex"


class StepStatus(str, Enum):
    """Planner step state."""

    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    skipped = "skipped"


class PlanStep(BaseModel):
    """One executable planning step."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    retrieve_memory: bool = False
    use_tools: bool = False
    ask_subquestions: bool = False
    evaluate_progress: bool = True
    status: StepStatus = StepStatus.pending
    attempts: int = 0
    max_attempts: int = 2
    notes: list[str] = Field(default_factory=list)


class TaskState(BaseModel):
    """Track a planned task."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_request: str
    complexity: TaskComplexity
    steps: list[PlanStep]
    current_step_index: int = 0
    max_total_attempts: int = 8
    completed: bool = False
    failed: bool = False


def detect_complexity(user_request: str) -> TaskComplexity:
    """Classify simple versus complex tasks with conservative heuristics."""

    lowered = user_request.lower()
    complex_markers = [
        "plan",
        "implement",
        "build",
        "compare",
        "research",
        "multiple",
        "workflow",
        "architecture",
        "debug",
        "fix",
    ]
    if len(user_request.split()) > 28 or any(marker in lowered for marker in complex_markers):
        return TaskComplexity.complex
    return TaskComplexity.simple


def generate_plan(user_request: str) -> TaskState:
    """Generate a bounded hierarchical plan."""

    complexity = detect_complexity(user_request)
    if complexity == TaskComplexity.simple:
        steps = [
            PlanStep(description="Retrieve relevant memory.", retrieve_memory=True),
            PlanStep(description="Answer directly and evaluate usefulness.", evaluate_progress=True),
        ]
    else:
        steps = [
            PlanStep(description="Clarify objective and constraints.", ask_subquestions=True),
            PlanStep(description="Retrieve relevant memory and prior lessons.", retrieve_memory=True),
            PlanStep(description="Draft a multi-step approach.", evaluate_progress=True),
            PlanStep(
                description="Identify tool needs without executing tools autonomously.",
                use_tools=True,
                evaluate_progress=True,
            ),
            PlanStep(description="Critique plan for safety, consistency, and gaps.", evaluate_progress=True),
            PlanStep(description="Synthesize final response.", evaluate_progress=True),
        ]
    return TaskState(user_request=user_request, complexity=complexity, steps=steps)


def advance_task(state: TaskState, step_success: bool, note: str = "") -> TaskState:
    """Advance task state with retries and loop limits."""

    total_attempts = sum(step.attempts for step in state.steps)
    if total_attempts >= state.max_total_attempts:
        state.failed = True
        state.completed = False
        return state

    if state.current_step_index >= len(state.steps):
        state.completed = True
        return state

    step = state.steps[state.current_step_index]
    step.attempts += 1
    if note:
        step.notes.append(note)

    if step_success:
        step.status = StepStatus.succeeded
        state.current_step_index += 1
    elif step.attempts < step.max_attempts:
        step.status = StepStatus.pending
        step.notes.append("Retry scheduled.")
    else:
        step.status = StepStatus.failed
        state.failed = True

    if state.current_step_index >= len(state.steps) and not state.failed:
        state.completed = True
    return state


def render_plan_debug(state: TaskState) -> str:
    """Render a compact markdown debug visualization."""

    lines = [
        f"### Plan Debug: {state.id}",
        f"- complexity: {state.complexity.value}",
        f"- completed: {state.completed}",
        f"- failed: {state.failed}",
        "",
        "| Step | Status | Capabilities | Attempts | Description |",
        "|---:|---|---|---:|---|",
    ]
    for index, step in enumerate(state.steps, start=1):
        capabilities = []
        if step.retrieve_memory:
            capabilities.append("memory")
        if step.use_tools:
            capabilities.append("tools-disabled")
        if step.ask_subquestions:
            capabilities.append("subquestions")
        if step.evaluate_progress:
            capabilities.append("evaluate")
        lines.append(
            f"| {index} | {step.status.value} | {', '.join(capabilities) or 'none'} | "
            f"{step.attempts}/{step.max_attempts} | {step.description} |"
        )
    return "\n".join(lines)
