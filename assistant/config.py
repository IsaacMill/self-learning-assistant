"""Application configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["openai", "openrouter", "ollama"]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_provider: ProviderName = Field(default="openai", alias="LLM_PROVIDER")
    llm_api_key: Optional[str] = Field(default=None, alias="LLM_API_KEY")
    llm_base_url: Optional[str] = Field(default=None, alias="LLM_BASE_URL")
    llm_model: Optional[str] = Field(default=None, alias="LLM_MODEL")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    database_path: Path = Field(default=Path("data/assistant.sqlite3"), alias="ASSISTANT_DB")
    chroma_path: Path = Field(default=Path("data/chroma"), alias="ASSISTANT_CHROMA_PATH")
    debug: bool = Field(default=False, alias="ASSISTANT_DEBUG")

    @property
    def resolved_model(self) -> str:
        """Return the configured model with provider-aware defaults."""

        if self.llm_model:
            return self.llm_model
        if self.llm_provider == "openrouter":
            return "openai/gpt-4o-mini"
        if self.llm_provider == "ollama":
            return "llama3.1"
        return self.openai_model

    @property
    def resolved_api_key(self) -> Optional[str]:
        """Return the configured API key, preserving old OPENAI_* env support."""

        if self.llm_api_key:
            return self.llm_api_key
        if self.llm_provider == "openai":
            return self.openai_api_key
        return None

    @property
    def resolved_base_url(self) -> Optional[str]:
        """Return the configured base URL with provider defaults."""

        if self.llm_base_url:
            return self.llm_base_url
        if self.llm_provider == "openai" and self.openai_base_url:
            return self.openai_base_url
        if self.llm_provider == "openrouter":
            return "https://openrouter.ai/api/v1"
        if self.llm_provider == "ollama":
            return "http://localhost:11434/v1"
        return None


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
