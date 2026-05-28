"""Simple CLI chat interface.

Run with:
    python -m assistant.chat
"""

from __future__ import annotations

from assistant.config import get_settings
from assistant.judge import consistency_judge, factuality_judge, safety_judge, usefulness_judge
from assistant.llm import LLMClient
from assistant.memory import MemoryStore
from assistant.reflection import reflect
from assistant.retrieval import MemoryRetriever
from assistant.safety import is_blocked_action_request, validate_memory_for_storage
from assistant.schemas import MemoryItem, MemoryKind
from assistant.self_analysis import run_self_analysis


def main() -> None:
    """Start an interactive local chat loop."""

    settings = get_settings()
    store = MemoryStore(settings.database_path)
    retriever = MemoryRetriever(store, settings.chroma_path)
    llm = LLMClient(
        api_key=settings.resolved_api_key,
        model=settings.resolved_model,
        base_url=settings.resolved_base_url,
        provider=settings.llm_provider,
    )
    debug = settings.debug

    print("Self-learning assistant MVP. Type /quit to exit, /debug to toggle memory debug.")
    while True:
        user_message = input("\nYou: ").strip()
        if user_message in {"/quit", "/exit"}:
            break
        if user_message == "/debug":
            debug = not debug
            print(f"Debug mode: {debug}")
            continue
        if user_message == "/self_analyze":
            print(run_self_analysis())
            continue
        if not user_message:
            continue
        if is_blocked_action_request(user_message):
            answer = (
                "I cannot execute shell commands, delete files, edit my own source, or perform autonomous tool use "
                "in this MVP. I can discuss a safe plan or explain the boundary."
            )
            judges = {
                "factuality": factuality_judge(answer, []),
                "safety": safety_judge(answer),
                "usefulness": usefulness_judge(user_message, answer),
                "consistency": consistency_judge(answer, []),
            }
            print(f"\nAssistant: {answer}")
            if debug:
                print("\nJudges:")
                for name, result in judges.items():
                    print(f"- {name}: pass={result.passed} score={result.score:.2f} {result.explanation}")
            store.log_interaction(user_message, answer, {name: result.model_dump() for name, result in judges.items()})
            continue

        used_memories = retriever.search(user_message, limit=5)
        if debug and used_memories:
            print("\nRetrieved memories:")
            for found in used_memories:
                print(f"- {found.memory.id[:8]} score={found.score:.2f}: {found.memory.content}")

        answer = llm.complete(user_message, used_memories)
        judges = {
            "factuality": factuality_judge(answer, used_memories),
            "safety": safety_judge(answer),
            "usefulness": usefulness_judge(user_message, answer),
            "consistency": consistency_judge(answer, used_memories),
        }

        print(f"\nAssistant: {answer}")
        if debug:
            print("\nJudges:")
            for name, result in judges.items():
                print(f"- {name}: pass={result.passed} score={result.score:.2f} {result.explanation}")

        reflection = reflect(user_message, answer)
        episodic = MemoryItem(
            kind=MemoryKind.episodic,
            content=reflection.summary,
            source="interaction",
            confidence=0.6,
            tags=["conversation"],
        )
        if validate_memory_for_storage(episodic).passed:
            store.add_memory(episodic)
        store.log_interaction(user_message, answer, {name: result.model_dump() for name, result in judges.items()})
        if reflection.possible_mistakes and debug:
            print("\nPossible mistakes:")
            for mistake in reflection.possible_mistakes:
                print(f"- {mistake}")

        for memory in reflection.proposed_memories:
            print(f"\nProposed trusted memory: {memory.content}")
            approved = input("Store as trusted semantic memory? [y/N]: ").strip().lower()
            if approved == "y":
                stored = store.add_memory(memory)
                promoted = store.promote_memory(stored.id, approved_by="cli-user")
                retriever.index_memory(promoted)
                print(f"Stored memory {promoted.id[:8]}.")
            else:
                print("Skipped memory.")


if __name__ == "__main__":
    main()
