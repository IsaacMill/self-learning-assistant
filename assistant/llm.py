"""OpenAI API-compatible LLM wrapper."""

from __future__ import annotations

from assistant.schemas import RetrievedMemory

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


class LLMClient:
    """Small wrapper around OpenAI-compatible chat completions."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str | None = None,
        provider: str = "openai",
    ):
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.base_url = self._resolve_base_url(provider, base_url)

    def complete(self, user_message: str, memories: list[RetrievedMemory]) -> str:
        """Return an assistant answer, using a local fallback without an API key."""

        context = self._format_memory_context(memories)
        if not self._has_remote_credentials():
            return self._fallback_answer(user_message, context)

        try:
            from openai import APIConnectionError, APIError, APIStatusError, OpenAI, RateLimitError

            client = OpenAI(api_key=self._client_api_key(), base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(user_message, context),
            )
            return response.choices[0].message.content or ""
        except RateLimitError:
            return self._provider_error_answer(
                "The configured LLM provider reported a rate limit or quota error. "
                "I can continue in local fallback mode, but remote model output is unavailable right now.",
                context,
            )
        except APIStatusError as exc:
            if exc.status_code in {402, 429}:
                return self._provider_error_answer(
                    "The configured LLM provider reported an account quota, billing, or rate-limit error. "
                    "I can continue in local fallback mode, but remote model output is unavailable right now.",
                    context,
                )
            return self._provider_error_answer(
                f"The configured LLM provider returned HTTP {exc.status_code}. "
                "I can continue in local fallback mode.",
                context,
            )
        except (APIConnectionError, APIError) as exc:
            return self._provider_error_answer(
                f"The configured LLM provider was unavailable: {exc.__class__.__name__}. "
                "I can continue in local fallback mode.",
                context,
            )
        except Exception as exc:
            return self._provider_error_answer(
                f"The configured LLM provider failed unexpectedly: {exc.__class__.__name__}. "
                "I can continue in local fallback mode.",
                context,
            )

    @staticmethod
    def _resolve_base_url(provider: str, base_url: str | None) -> str | None:
        if base_url:
            return base_url
        if provider == "openrouter":
            return OPENROUTER_BASE_URL
        if provider == "ollama":
            return OLLAMA_BASE_URL
        return None

    def _client_api_key(self) -> str:
        if self.provider == "ollama":
            return self.api_key or "ollama"
        return self.api_key or ""

    def _has_remote_credentials(self) -> bool:
        if self.provider == "ollama":
            return True
        return bool(self.api_key)

    @staticmethod
    def _build_messages(user_message: str, context: str) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a safe local personal assistant MVP. Use retrieved memory only when relevant. "
                    "Do not claim to run commands, browse the internet, delete files, or edit your own code."
                ),
            },
            {"role": "system", "content": context or "No relevant memory was retrieved."},
            {"role": "user", "content": user_message},
        ]
        return messages

    @staticmethod
    def _format_memory_context(memories: list[RetrievedMemory]) -> str:
        if not memories:
            return ""
        lines = ["Relevant memories:"]
        for item in memories:
            lines.append(f"- [{item.memory.kind.value}/{item.memory.trust.value}] {item.memory.content}")
        return "\n".join(lines)

    @staticmethod
    def _fallback_answer(user_message: str, context: str) -> str:
        prefix = "I am running in local fallback mode without an LLM API key."
        if context:
            return f"{prefix} Relevant memory context:\n{context}\n\nI can use this context to respond safely."
        return f"{prefix} I can still log, retrieve, judge, and reflect on this message safely."

    @staticmethod
    def _provider_error_answer(message: str, context: str) -> str:
        if context:
            return f"{message}\n\nRelevant memory context was still retrieved:\n{context}"
        return message
