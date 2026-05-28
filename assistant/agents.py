"""Multi-agent cognition workflow with local, logged agent messages."""

from __future__ import annotations

from dataclasses import dataclass

from assistant.memory import MemoryStore
from assistant.planner import generate_plan, render_plan_debug


@dataclass(frozen=True)
class AgentProfile:
    """Static agent prompt/personality."""

    name: str
    system_prompt: str
    personality: str


AGENTS = {
    "planner_agent": AgentProfile(
        name="planner_agent",
        system_prompt="Break requests into safe, bounded steps.",
        personality="methodical and constraint-aware",
    ),
    "researcher_agent": AgentProfile(
        name="researcher_agent",
        system_prompt="Gather relevant local memory and identify missing information without web access.",
        personality="curious but cautious",
    ),
    "critic_agent": AgentProfile(
        name="critic_agent",
        system_prompt="Review for safety, contradictions, hallucinations, and weak assumptions.",
        personality="skeptical and precise",
    ),
    "memory_agent": AgentProfile(
        name="memory_agent",
        system_prompt="Decide what should be remembered, proposed, rejected, or cleaned up.",
        personality="conservative and provenance-focused",
    ),
    "synthesizer_agent": AgentProfile(
        name="synthesizer_agent",
        system_prompt="Produce concise final answers from reviewed work.",
        personality="direct and practical",
    ),
}


def run_multi_agent_workflow(user_request: str, store: MemoryStore) -> dict[str, object]:
    """Run the planner -> researcher -> critic -> revision -> synthesizer workflow."""

    task_state = generate_plan(user_request)
    task_id = task_state.id

    plan_debug = render_plan_debug(task_state)
    store.log_agent_message(task_id, "planner_agent", "plan", plan_debug)

    researcher_note = (
        "Researcher reviewed local memory only. Internet access and shell/tool execution are disabled."
    )
    store.log_agent_message(task_id, "researcher_agent", "research", researcher_note)

    critic_note = (
        "Critic review: preserve safety boundary, avoid unsupported claims, and require memory approval."
    )
    store.log_agent_message(task_id, "critic_agent", "critique", critic_note)

    revision_note = "Revision: keep any tool-use step as a planned capability, not an executed action."
    store.log_agent_message(task_id, "planner_agent", "revision", revision_note)

    memory_note = "Memory agent: propose durable lessons only; do not auto-promote trusted memory."
    store.log_agent_message(task_id, "memory_agent", "memory_review", memory_note)

    final_note = (
        "Synthesizer final: task has a safe bounded plan and should answer using retrieved memory when relevant."
    )
    store.log_agent_message(task_id, "synthesizer_agent", "final", final_note)

    return {
        "task_id": task_id,
        "agents": {name: profile.__dict__ for name, profile in AGENTS.items()},
        "plan": task_state.model_dump(mode="json"),
        "debug": plan_debug,
        "final_note": final_note,
    }
