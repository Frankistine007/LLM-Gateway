import json
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from sqlalchemy.orm import Session

from app import models
from app.dependencies import get_current_client, get_db
from app.providers.errors import is_retryable
from app.providers.registry import ROUTING_TIERS, resolve_fallback, resolve_provider
from app.providers.text import extract_text
from app.services.make_cache_key import make_cache_key
from app.services.cache_store import get_cached_response
from app.services.cache_store import set_cached_response

from app.services.circuit_breaker import (
    CircuitOpenError,
    is_open,
    record_failure,
    record_success,
)

from app.services.rate_limit import (
    estimate_tokens,
    rate_limit_headers,
    reconcile_tokens,
    reserve_capacity,
)
from app.services.routing import classify_prompt

router = APIRouter(tags=["chat"])


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: Request,
    client: models.Client = Depends(get_current_client),
    db: Session = Depends(get_db),
):
    body = await request.json()
    messages = body.get("messages", [])
    requested_model = body.get("model")
    stream = body.get("stream", True)
    max_tokens = body.get("max_tokens")
    

    # No model, or explicit "auto": guess a tier from the prompt and route to
    # it. Anything else is an explicit caller choice and is always honored —
    # silently overriding a caller's stated model would break their
    # expectations without warning.

    if requested_model in (None, "auto"):
        tier = classify_prompt(messages)
        model = ROUTING_TIERS[tier]
        routed_by = f"auto:{tier}"
    else:
        model = requested_model
        routed_by = "explicit"

    estimated_tokens = estimate_tokens(messages, max_tokens)
    reserve_capacity(client, db, estimated_tokens)
    headers = rate_limit_headers(client)
    headers["X-Gateway-Routing"] = routed_by
    headers["X-Gateway-Model"] = model
    cache_key = make_cache_key(
        client.id,
        model,
        messages,
        body.get("temperature"),
        max_tokens,
    )
    cached_text = get_cached_response(cache_key)

    if cached_text is not None:
        headers["X-Gateway-Cache"] = "HIT"
        db.add(
            models.RequestLog(
                client_id=client.id,
                prompt=json.dumps(messages),
                model=model,
                response=cached_text,
            )
        )
        db.commit()

        if stream:
            async def cached_stream():
                yield cached_text

            return StreamingResponse(
                cached_stream(),
                media_type="text/plain",
                headers=headers,
            )

        return PlainTextResponse(cached_text, headers=headers)
    
    else: 
        headers["X-Gateway-Cache"] = "MISS"
    
    provider_name, llm = resolve_provider(model)

    log = models.RequestLog(
        client_id=client.id,
        prompt=json.dumps(messages),
        provider=provider_name,
        model=model,
        response=cached_text,
    )

    lc_messages = [(m["role"], m["content"]) for m in messages]

    if not stream:
        return await _handle_non_streaming(
            lc_messages, provider_name, llm, log, db, client, estimated_tokens, headers,cache_key
        )

    return StreamingResponse(
        _token_stream(lc_messages, provider_name, llm, log, db, client, estimated_tokens, cache_key),
        media_type="text/plain",
        headers=headers,
    )


async def _handle_non_streaming(
    lc_messages, provider_name, llm, log, db: Session, client, estimated_tokens: int, headers, cache_key: str
):
    start = time.perf_counter()
    try:
        try:
            if is_open(db, provider_name):
                raise CircuitOpenError(provider_name)
            result = await llm.ainvoke(lc_messages)
            record_success(db, provider_name)
        except Exception as primary_exc:
            circuit_open = isinstance(primary_exc, CircuitOpenError)
            retryable = circuit_open or is_retryable(primary_exc)
            if not circuit_open and is_retryable(primary_exc):
                record_failure(db, provider_name)

            fallback = resolve_fallback(provider_name)
            if not retryable or fallback is None:
                raise

            fb_name, fb_model, fb_llm = fallback
            try:
                if is_open(db, fb_name):
                    raise CircuitOpenError(fb_name)
                result = await fb_llm.ainvoke(lc_messages)
                record_success(db, fb_name)
            except Exception as fallback_exc:
                if not isinstance(fallback_exc, CircuitOpenError) and is_retryable(
                    fallback_exc
                ):
                    record_failure(db, fb_name)
                raise

            log.provider, log.model = fb_name, fb_model
            log.error_message = (
                f"{provider_name} unavailable (circuit open); fell back to {fb_name}"
                if circuit_open
                else f"{provider_name} failed ({primary_exc}); fell back to {fb_name}"
            )

        text = extract_text(result.content)
        
        set_cached_response(cache_key, text)

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


async def _token_stream(lc_messages, provider_name, llm, log, db: Session, client, estimated_tokens: int, cache_key: str):
    start = time.perf_counter()
    full_response = ""
    sent_any = False
    completed = False

    async def consume(model_llm):
        nonlocal full_response, sent_any
        async for chunk in model_llm.astream(lc_messages):
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
            if is_open(db, provider_name):
                raise CircuitOpenError(provider_name)
            async for text in consume(llm):
                yield text
            record_success(db, provider_name)
        except Exception as primary_exc:
            circuit_open = isinstance(primary_exc, CircuitOpenError)
            retryable = circuit_open or is_retryable(primary_exc)
            if not circuit_open and is_retryable(primary_exc):
                record_failure(db, provider_name)

            fallback = resolve_fallback(provider_name)
            if not retryable or fallback is None:
                raise

            fb_name, fb_model, fb_llm = fallback
            log.provider, log.model = fb_name, fb_model
            log.error_message = (
                f"{provider_name} unavailable (circuit open); fell back to {fb_name}"
                if circuit_open
                else f"{provider_name} failed mid-stream ({primary_exc}); "
                f"fell back to {fb_name}"
            )

            notice = (
                f"\n\n[gateway] '{provider_name}' "
                f"{'is unavailable' if circuit_open else 'failed'}"
                f"{' mid-response' if sent_any else ''}; switched to "
                f"'{fb_name}'."
                f"{' Text before and after this point may be inconsistent.' if sent_any else ''}"
                "\n\n"
            )
            yield notice
            full_response += notice
    
            try:
                if is_open(db, fb_name):
                    raise CircuitOpenError(fb_name)
                async for text in consume(fb_llm):
                    yield text
                record_success(db, fb_name)
            except Exception as fallback_exc:
                if not isinstance(fallback_exc, CircuitOpenError) and is_retryable(
                    fallback_exc
                ):
                    record_failure(db, fb_name)
                raise

        completed = True
            
    except Exception as exc:
        log.error_message = str(exc)
    finally:
        log.response = full_response
        log.latency = time.perf_counter() - start
        db.add(log)
        db.commit()
        if completed:
            set_cached_response(cache_key, full_response)
        reconcile_tokens(client, db, estimated_tokens, log.tokens_used)
