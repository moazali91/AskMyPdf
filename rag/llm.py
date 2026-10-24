"""LLM factory: Ollama (offline) or Groq (online), plus readable error messages."""

import json
import urllib.request

from langchain_core.language_models.chat_models import BaseChatModel

from rag import config

OLLAMA = "Ollama (offline)"
GROQ = "Groq (online)"


class LLMError(Exception):
    """An LLM problem with a message that is safe to show the user."""


def available_providers() -> list[str]:
    return [GROQ] if config.ON_CLOUD else [OLLAMA, GROQ]


def model_label(provider: str) -> str:
    """Display label, e.g. "Groq · gpt-oss-20b" (the "openai/" prefix is just the
    model's publisher on Groq's catalogue; requests still go only to Groq)."""
    if provider == OLLAMA:
        return f"Ollama · {config.OLLAMA_MODEL}"
    return f"Groq · {config.GROQ_MODEL.split('/')[-1]}"


def provider_ready(provider: str) -> tuple[bool, str]:
    """Quick health check for the sidebar: (ok, short status text)."""
    if provider == GROQ:
        return (True, "Ready") if config.GROQ_API_KEY else (False, "No API key")
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=1) as r:
            names = {m["name"] for m in json.load(r).get("models", [])}
    except Exception:
        return False, "Ollama not running"
    if config.OLLAMA_MODEL not in names:
        return False, "Model not downloaded"
    return True, "Running locally"


def warm_up_ollama() -> None:
    """Load the Ollama model into memory so the first question isn't slow.

    An empty prompt makes Ollama load the model without generating anything.
    Errors are ignored: the sidebar status already reports a stopped server.
    """
    body = json.dumps(
        {"model": config.OLLAMA_MODEL, "prompt": "", "keep_alive": config.OLLAMA_KEEP_ALIVE}
    ).encode()
    req = urllib.request.Request(
        f"{config.OLLAMA_BASE_URL}/api/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=120).read()
    except Exception:
        pass


def complete(provider: str, messages: list) -> str:
    """Non-streaming call for short helper tasks (e.g. rewriting a follow-up)."""
    try:
        return get_llm(provider).invoke(messages).content
    except Exception as err:
        raise friendly_error(err, provider) from err


def get_llm(provider: str) -> BaseChatModel:
    if provider == OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.TEMPERATURE,
            keep_alive=config.OLLAMA_KEEP_ALIVE,
        )

    if provider == GROQ:
        if not config.GROQ_API_KEY:
            raise LLMError(
                "No Groq API key found. Add GROQ_API_KEY to your .env file "
                "(or to Streamlit secrets when deployed)."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=config.TEMPERATURE,
            reasoning_effort=config.GROQ_REASONING_EFFORT,
            max_retries=1,
        )

    raise LLMError(f"Unknown model provider: {provider}")


def friendly_error(err: Exception, provider: str) -> LLMError:
    """Turn a raw client exception into a message the user can act on."""
    if isinstance(err, LLMError):
        return err
    name = type(err).__name__
    text = str(err)

    if provider == OLLAMA:
        if "ConnectError" in name or "Connection" in name or "refused" in text.lower():
            return LLMError(
                "Can't reach Ollama at "
                f"{config.OLLAMA_BASE_URL}. Start the Ollama app or run `ollama serve`."
            )
        if "not found" in text.lower():
            return LLMError(
                f"Ollama model '{config.OLLAMA_MODEL}' isn't downloaded. "
                f"Run `ollama pull {config.OLLAMA_MODEL}`."
            )

    if provider == GROQ:
        if name == "RateLimitError" or "429" in text:
            return LLMError(
                "Groq's free rate limit was reached. Wait a minute and try again, "
                "or switch to Ollama."
            )
        if name == "AuthenticationError" or "401" in text:
            return LLMError("Groq rejected the API key. Check GROQ_API_KEY.")
        if name == "NotFoundError" or "model_not_found" in text:
            return LLMError(
                f"Groq model '{config.GROQ_MODEL}' is unavailable. Pick a current model "
                "from console.groq.com/docs/models and update GROQ_MODEL in rag/config.py."
            )
        if "Connection" in name:
            return LLMError("Can't reach Groq. Check your internet connection.")

    return LLMError(f"{provider} error: {name}: {text}")
