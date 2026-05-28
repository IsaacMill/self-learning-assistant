import openai
import httpx

from assistant.config import get_settings
from assistant.llm import LLMClient, OPENROUTER_BASE_URL


def test_openrouter_defaults_to_required_base_url() -> None:
    client = LLMClient(
        api_key="test-key",
        model="openai/gpt-4o-mini",
        provider="openrouter",
    )

    assert client.base_url == OPENROUTER_BASE_URL


def test_explicit_base_url_overrides_provider_default() -> None:
    client = LLMClient(
        api_key="test-key",
        model="some-model",
        base_url="https://example.test/v1",
        provider="openrouter",
    )

    assert client.base_url == "https://example.test/v1"


def test_settings_resolve_openrouter_defaults(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.resolved_base_url == OPENROUTER_BASE_URL
    assert settings.resolved_model == "openai/gpt-4o-mini"


def test_llm_provider_errors_return_graceful_fallback(monkeypatch) -> None:
    class BrokenCompletions:
        def create(self, **kwargs):
            raise RuntimeError("provider unavailable")

    class BrokenChat:
        completions = BrokenCompletions()

    class BrokenOpenAI:
        def __init__(self, **kwargs):
            self.chat = BrokenChat()

    monkeypatch.setattr(openai, "OpenAI", BrokenOpenAI)
    client = LLMClient(api_key="test-key", model="test-model", provider="openai")

    answer = client.complete("hello", [])

    assert "failed unexpectedly" in answer
    assert "fallback mode" in answer


def test_rate_limit_status_returns_graceful_fallback(monkeypatch) -> None:
    class RateLimitedCompletions:
        def create(self, **kwargs):
            response = httpx.Response(
                status_code=429,
                request=httpx.Request("POST", "https://example.test/v1/chat/completions"),
            )
            raise openai.APIStatusError("rate limited", response=response, body=None)

    class RateLimitedChat:
        completions = RateLimitedCompletions()

    class RateLimitedOpenAI:
        def __init__(self, **kwargs):
            self.chat = RateLimitedChat()

    monkeypatch.setattr(openai, "OpenAI", RateLimitedOpenAI)
    client = LLMClient(api_key="test-key", model="test-model", provider="openrouter")

    answer = client.complete("hello", [])

    assert "quota" in answer
    assert "rate-limit" in answer
    assert "fallback mode" in answer
