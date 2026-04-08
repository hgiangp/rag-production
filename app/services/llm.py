"""LLM factory — provider-agnostic, with retry and optional Langfuse tracing."""

from typing import Any, Literal, Optional

import structlog
from langchain_core.language_models import BaseChatModel
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = structlog.get_logger(__name__)

_instances: dict[str, Any] = {}


def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    framework: Literal["langchain", "llamaindex"] = "langchain",
) -> Any:
    """Return a cached LLM instance for the given model.

    Attaches Langfuse callback handler automatically if configured.
    """
    model = model or settings.DEFAULT_LLM_MODEL
    temperature = temperature if temperature is not None else settings.DEFAULT_LLM_TEMPERATURE
    cache_key = f"{framework}:{model}:{temperature}"

    if cache_key not in _instances:
        _instances[cache_key] = _build_llm(model=model, temperature=temperature, framework=framework)

    return _instances[cache_key]


def _build_llm(model: str, temperature: float, framework: str) -> Any:
    provider = settings.LLM_PROVIDER

    if framework == "llamaindex":
        return _build_llamaindex_llm(provider=provider, model=model, temperature=temperature)
    return _build_langchain_llm(provider=provider, model=model, temperature=temperature)


def _build_langchain_llm(provider: str, model: str, temperature: float) -> BaseChatModel:
    callbacks = []
    if settings.langfuse_enabled:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
        # Langfuse singleton must be registered before CallbackHandler.__init__ calls get_client()
        Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        callbacks.append(CallbackHandler(public_key=settings.LANGFUSE_PUBLIC_KEY))

    match provider:
        case "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=settings.MAX_TOKENS,
                openai_api_key=settings.OPENAI_API_KEY,
                base_url=settings.LLM_BASE_URL or None,
                callbacks=callbacks or None,
            )
        case "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model_name=model,
                temperature=temperature,
                max_tokens=settings.MAX_TOKENS,
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                anthropic_api_url=settings.ANTHROPIC_BASE_URL or None,
                callbacks=callbacks or None,
            )
        case "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model,
                temperature=temperature,
                base_url=settings.OLLAMA_LLM_BASE_URL,
                callbacks=callbacks or None,
            )
        case _:
            raise ValueError(f"Unknown LLM provider: {provider}")


def _build_llamaindex_llm(provider: str, model: str, temperature: float) -> Any:
    match provider:
        case "openai":
            from llama_index.llms.openai import OpenAI
            return OpenAI(
                model=model,
                temperature=temperature,
                api_key=settings.OPENAI_API_KEY,
                api_base=settings.LLM_BASE_URL or None,
            )
        case "anthropic":
            from llama_index.llms.anthropic import Anthropic
            return Anthropic(
                model=model,
                temperature=temperature,
                api_key=settings.ANTHROPIC_API_KEY,
                base_url=settings.ANTHROPIC_BASE_URL or None,
            )
        case "ollama":
            from llama_index.llms.ollama import Ollama
            return Ollama(model=model, temperature=temperature, base_url=settings.OLLAMA_LLM_BASE_URL)
        case _:
            raise ValueError(f"Unknown LLM provider for LlamaIndex: {provider}")


# Singleton for default inference LLM
llm_service = get_llm()
