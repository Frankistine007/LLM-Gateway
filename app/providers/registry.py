from fastapi import HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from app.config import settings


def build_groq(model: str):
    return ChatGroq(model=model, api_key=settings.groq_api_key)


def build_gemini(model: str):
    return ChatGoogleGenerativeAI(model=model, google_api_key=settings.google_api_key)


# model name -> (provider name, factory). The factory defers construction so a
# model is only instantiated when it is actually routed to.
# Groq retired the llama-3.1/3.3 line (confirmed via GET /v1/models against
# the real API on 2026-08-17); openai/gpt-oss-20b and -120b are the current
# small/large pair.
MODEL_REGISTRY = {
    "openai/gpt-oss-20b": ("groq", build_groq),
    "openai/gpt-oss-120b": ("groq", build_groq),
    "gemini-flash-latest": ("gemini", build_gemini),
    "gemini-flash-lite-latest": ("gemini", build_gemini),
    "gemini-2.0-flash": ("gemini", build_gemini),
}

DEFAULT_MODEL = "openai/gpt-oss-20b"

# Where classify_prompt()'s tier guess lands when the caller doesn't name a
# model explicitly (or asks for "auto"). Both tiers stay on Groq for now —
# crossing providers here would conflate routing with the fallback system.
ROUTING_TIERS = {
    "simple": "openai/gpt-oss-20b",
    "complex": "openai/gpt-oss-120b",
}

# Where to send a request when its primary provider fails with a retryable error.
FALLBACK_MODEL = {
    "groq": "gemini-flash-lite-latest",
    "gemini": "openai/gpt-oss-20b",
}


def resolve_provider(model: str):
    """Return (provider_name, llm) for a requested model."""
    entry = MODEL_REGISTRY.get(model)
    if entry is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: "
            f"{', '.join(sorted(MODEL_REGISTRY))}",
        )

    provider_name, factory = entry
    return provider_name, factory(model)


def resolve_fallback(provider_name: str):
    """Return (provider_name, model, llm) to fall back to, or None."""
    fallback_model = FALLBACK_MODEL.get(provider_name)
    if fallback_model is None or fallback_model not in MODEL_REGISTRY:
        return None

    name, llm = resolve_provider(fallback_model)
    return name, fallback_model, llm
