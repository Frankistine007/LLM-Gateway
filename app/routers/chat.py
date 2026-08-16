import json
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from sqlalchemy.orm import Session

from app import models
from app.dependencies import get_current_client, get_db
from app.providers.errors import is_retryable
from app.providers.registry import DEFAULT_MODEL, resolve_fallback, resolve_provider
from app.providers.text import extract_text
from app.services.rate_limit import (
    estimate_tokens,
    rate_limit_headers,
    reconcile_tokens,
    reserve_capacity,
)

router = APIRouter(tags=["chat"])


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: Request,
    client: models.Client = Depends(get_current_client),
    db: Session = Depends(get_db),
):
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", DEFAULT_MODEL)
    stream = body.get("stream", True)
    max_tokens = body.get("max_tokens")

    estimated_tokens = estimate_tokens(messages, max_tokens)
    reserve_capacity(client, db, estimated_tokens)
    headers = rate_limit_headers(client)

    provider_name, llm = resolve_provider(model)

    log = models.RequestLog(
        client_id=client.id,
        prompt=json.dumps(messages),
        provider=provider_name,
        model=model,
    )

    lc_messages = [(m["role"], m["content"]) for m in messages]

    if not stream:
        return _handle_non_streaming(
            lc_messages, provider_name, llm, log, db, client, estimated_tokens, headers
        )

    return StreamingResponse(
        _token_stream(lc_messages, provider_name, llm, log, db, client, estimated_tokens),
        media_type="text/plain",
        headers=headers,
    )


def _handle_non_streaming(
    lc_messages, provider_name, llm, log, db: Session, client, estimated_tokens: int, headers
):
    start = time.perf_counter()
    try:
        try:
            result = llm.invoke(lc_messages)
        except Exception as primary_exc:
            fallback = resolve_fallback(provider_name)
            if not is_retryable(primary_exc) or fallback is None:
                raise

            fb_name, fb_model, fb_llm = fallback
            result = fb_llm.invoke(lc_messages)
            log.provider, log.model = fb_name, fb_model
            log.error_message = (
                f"{provider_name} failed ({primary_exc}); fell back to {fb_name}"
            )

        text = extract_text(result.content)
        log.response = text
        log.tokens_used = (result.usage_metadata or {}).get("total_tokens")
        return PlainTextResponse(text, headers=headers)
    except Exception as exc:
        log.error_message = str(exc)
        raise HTTPException(status_code=502, detail=str(exc), headers=headers) from exc
    finally:
        log.latency = time.perf_counter() - start
        db.add(log)
        db.commit()
        reconcile_tokens(client, db, estimated_tokens, log.tokens_used)


def _token_stream(lc_messages, provider_name, llm, log, db: Session, client, estimated_tokens: int):
    start = time.perf_counter()
    full_response = ""
    sent_any = False

    def consume(model_llm):
        nonlocal full_response, sent_any
        for chunk in model_llm.stream(lc_messages):
            text = extract_text(chunk.content)
            if chunk.usage_metadata:
                log.tokens_used = (
                    chunk.usage_metadata.get("total_tokens") or log.tokens_used
                )
            if text:
                full_response += text
                sent_any = True
                yield text

    try:
        try:
            yield from consume(llm)
        except Exception as primary_exc:
            fallback = resolve_fallback(provider_name)
            if not is_retryable(primary_exc) or fallback is None:
                raise

            fb_name, fb_model, fb_llm = fallback
            log.provider, log.model = fb_name, fb_model
            log.error_message = (
                f"{provider_name} failed mid-stream ({primary_exc}); "
                f"fell back to {fb_name}"
            )

            notice = (
                f"\n\n[gateway] '{provider_name}' failed"
                f"{' mid-response' if sent_any else ''}; switched to "
                f"'{fb_name}'."
                f"{' Text before and after this point may be inconsistent.' if sent_any else ''}"
                "\n\n"
            )
            yield notice
            full_response += notice

            yield from consume(fb_llm)
    except Exception as exc:
        log.error_message = str(exc)
    finally:
        log.response = full_response
        log.latency = time.perf_counter() - start
        db.add(log)
        db.commit()
        reconcile_tokens(client, db, estimated_tokens, log.tokens_used)
