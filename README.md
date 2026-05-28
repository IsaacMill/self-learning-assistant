# Self-Learning Assistant MVP

Local Python MVP for a safe, modular personal assistant with memory, retrieval, reflection, and judges.

## What It Does

- Stores episodic, semantic, and failure memories in SQLite.
- Requires every memory to include source, timestamp, confidence, tags, and trust state.
- Retrieves relevant non-raw memories before answering.
- Uses an OpenAI API-compatible chat wrapper with `openai`, `openrouter`, or `ollama`.
- Falls back to deterministic local responses without an API key.
- Runs factuality, safety, usefulness, consistency, and memory-write judges.
- Reflects after each interaction and proposes memory updates.
- Requires human approval before promoting memories to trusted semantic memory.

## Safety Boundary

This MVP intentionally does not implement autonomous internet access, shell execution, file deletion, runtime source editing, self-modifying code, or tool use. It also blocks likely secrets from memory and rejects sensitive personal information unless explicitly allowed by the caller.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Configure an LLM provider in `.env`. Without a remote API key, the assistant still exercises memory, retrieval, judges, and reflection in local fallback mode.

## LLM Providers

Provider-neutral settings:

```env
LLM_PROVIDER=openai
LLM_API_KEY=<your-api-key>
LLM_BASE_URL=
LLM_MODEL=gpt-4o-mini
```

OpenRouter:

```env
LLM_PROVIDER=openrouter
LLM_API_KEY=<your-openrouter-api-key>
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini
```

If `LLM_PROVIDER=openrouter` and `LLM_BASE_URL` is omitted, the assistant defaults to `https://openrouter.ai/api/v1`.

Ollama using its OpenAI-compatible endpoint:

```env
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1
```

The older `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` variables are still supported for compatibility. Quota, billing, rate-limit, and provider connectivity errors are handled gracefully by returning a safe fallback response instead of crashing the CLI or API.

## CLI

```bash
python -m assistant.chat
```

Commands:

- `/debug` toggles retrieved-memory and judge output.
- `/self_analyze` prints the latest improvement report from eval history.
- `/quit` exits.

## Memory Maintenance

Memories include usefulness, confidence, recency, and source-type metadata. Duplicate memories are merged, simple contradictions are flagged, and retrieval ranking prefers high-confidence, useful, recent memories.

Run cleanup to decay low-quality memories and reject weak untrusted memories:

```bash
python -m assistant.memory_cleanup
```

## Planning And Agents

The planner detects simple versus complex tasks, creates bounded multi-step plans, tracks step status, retries failed steps, and stops before infinite loops. Planner steps can declare that they need memory retrieval, tool use, subquestions, and progress evaluation; tool use remains disabled in this MVP.

The multi-agent workflow is implemented as a local, logged cognition layer:

```text
user request -> planner_agent -> researcher_agent -> critic_agent -> revisions -> memory_agent -> synthesizer_agent
```

All inter-agent communication is logged in SQLite.

## Evals

Run the benchmark suite:

```bash
python -m assistant.eval
```

The eval framework includes 50 benchmark tasks across reasoning, memory, coding, safety, planning, and consistency. It performs automatic criteria scoring, judge scoring, hallucination detection, failure lessons, future eval proposals, historical run snapshots in `evals/history/`, and a markdown leaderboard at `evals/leaderboard.md`.

Generate the latest improvement report:

```bash
python -m assistant.self_analysis
```

## API

```bash
uvicorn assistant.main:app --reload
```

Then POST to `/chat`:

```json
{
  "message": "Remember that I prefer concise answers.",
  "debug": true
}
```

## Architecture

- `assistant/main.py`: FastAPI app.
- `assistant/chat.py`: CLI chat loop.
- `assistant/llm.py`: OpenAI-compatible LLM wrapper.
- `assistant/memory.py`: SQLite memory and interaction logs.
- `assistant/retrieval.py`: ChromaDB retrieval with lexical fallback.
- `assistant/judge.py`: MVP heuristic judges.
- `assistant/reflection.py`: Post-interaction learning proposals.
- `assistant/schemas.py`: Pydantic models.
- `assistant/config.py`: Environment-backed settings.
- `assistant/safety.py`: Safety and memory-storage guards.

## Future TODOs

- Add approved web search with source attribution.
- Add explicit, permissioned tool use.
- Add robot integration behind a hardware safety layer.
- Add LoRA fine-tuning experiments from curated, human-approved data.
- Add richer embeddings and stronger LLM-based judges.
