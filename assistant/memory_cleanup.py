"""Memory cleanup job.

Run with:
    python -m assistant.memory_cleanup
"""

from __future__ import annotations

from assistant.config import get_settings
from assistant.memory import MemoryStore


def run_cleanup(min_quality: float = 0.35) -> dict[str, int]:
    """Run memory decay and rejection cleanup."""

    settings = get_settings()
    store = MemoryStore(settings.database_path)
    return store.cleanup_memories(min_quality=min_quality)


def main() -> None:
    """CLI entrypoint."""

    result = run_cleanup()
    print(f"Memory cleanup complete: decayed={result['decayed']} rejected={result['rejected']}")


if __name__ == "__main__":
    main()
