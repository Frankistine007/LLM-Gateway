from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app import models

WINDOW_SECONDS = 60

# No real max_tokens param exists on the request yet, so a fixed estimate stands
# in for "expected response size" until the caller can specify one.
CHARS_PER_TOKEN_ESTIMATE = 4
DEFAULT_RESPONSE_TOKEN_ESTIMATE = 512


def estimate_tokens(messages: list[dict], max_tokens: int | None = None) -> int:
    prompt_chars = sum(len(m.get("content", "") or "") for m in messages)
    prompt_estimate = max(1, prompt_chars // CHARS_PER_TOKEN_ESTIMATE)
    return prompt_estimate + (max_tokens or DEFAULT_RESPONSE_TOKEN_ESTIMATE)


def _refill(client: models.Client, now: datetime):
    """Lazily bring both buckets up to date. No background timer: the bucket
    level is recomputed from elapsed time on every read, which is what keeps
    this O(1) instead of scanning request history.
    """
    if client.bucket_updated_at is None:
        client.bucket_requests = float(client.rate_limit)
        client.bucket_tokens = float(client.token_limit)
        client.bucket_updated_at = now
        return

    # SQLite round-trips DateTime(timezone=True) as naive, even though it was
    # written tz-aware, so normalize before subtracting.
    last_updated = client.bucket_updated_at
    if last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=timezone.utc)

    elapsed = (now - last_updated).total_seconds()
    if elapsed <= 0:
        return

    request_refill_rate = client.rate_limit / WINDOW_SECONDS
    token_refill_rate = client.token_limit / WINDOW_SECONDS

    client.bucket_requests = min(
        client.rate_limit, client.bucket_requests + elapsed * request_refill_rate
    )
    client.bucket_tokens = min(
        client.token_limit, client.bucket_tokens + elapsed * token_refill_rate
    )
    client.bucket_updated_at = now


def rate_limit_headers(client: models.Client) -> dict[str, str]:
    """OpenAI-style rate limit headers describing the client's current buckets.

    Reports state as of the reservation, not settled usage: a streaming
    response must send headers before its body, which is long before real
    token usage is known, so both paths report the same (slightly
    pessimistic) post-reserve numbers rather than disagreeing.
    """
    request_refill_rate = client.rate_limit / WINDOW_SECONDS
    token_refill_rate = client.token_limit / WINDOW_SECONDS

    requests_remaining = max(0.0, client.bucket_requests)
    tokens_remaining = max(0.0, client.bucket_tokens)

    return {
        "X-RateLimit-Limit-Requests": str(client.rate_limit),
        "X-RateLimit-Remaining-Requests": str(int(requests_remaining)),
        "X-RateLimit-Reset-Requests": str(
            _seconds_until(client.bucket_requests, client.rate_limit, request_refill_rate)
        ),
        "X-RateLimit-Limit-Tokens": str(client.token_limit),
        "X-RateLimit-Remaining-Tokens": str(int(tokens_remaining)),
        "X-RateLimit-Reset-Tokens": str(
            _seconds_until(client.bucket_tokens, client.token_limit, token_refill_rate)
        ),
    }


def _seconds_until(level: float, target: float, refill_rate: float) -> int:
    """Whole seconds until a bucket refills from `level` back up to `target`."""
    if level >= target:
        return 0
    return max(1, int((target - level) / refill_rate) + 1)


def reserve_capacity(client: models.Client, db: Session, estimated_tokens: int):
    """Refills both buckets, then debits 1 request + the estimated token cost
    up front. Rejects with 429 before any provider call is made if either
    bucket can't cover the cost. Caller must follow a successful reserve with
    reconcile_tokens() once real usage is known.
    """
    now = datetime.now(timezone.utc)
    _refill(client, now)

    if client.bucket_requests < 1:
        db.add(client)
        db.commit()
        request_refill_rate = client.rate_limit / WINDOW_SECONDS
        wait = (1 - client.bucket_requests) / request_refill_rate
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {client.rate_limit} requests per "
            f"{WINDOW_SECONDS}s",
            headers={
                **rate_limit_headers(client),
                "Retry-After": str(max(1, int(wait) + 1)),
            },
        )

    if client.bucket_tokens < estimated_tokens:
        db.add(client)
        db.commit()
        token_refill_rate = client.token_limit / WINDOW_SECONDS
        wait = (estimated_tokens - client.bucket_tokens) / token_refill_rate
        raise HTTPException(
            status_code=429,
            detail=f"Token limit exceeded: {client.token_limit} tokens per "
            f"{WINDOW_SECONDS}s",
            headers={
                **rate_limit_headers(client),
                "Retry-After": str(max(1, int(wait) + 1)),
            },
        )

    client.bucket_requests -= 1
    client.bucket_tokens -= estimated_tokens
    db.add(client)
    db.commit()


def reconcile_tokens(
    client: models.Client, db: Session, estimated_tokens: int, actual_tokens: int | None
):
    """Corrects the token bucket once real usage is known. Refunds an
    overestimate, debits further on an underestimate. If actual usage is
    unknown (e.g. a stream that died before usage_metadata arrived), the
    estimate is left charged rather than refunded.
    """
    if actual_tokens is None:
        return

    client.bucket_tokens -= actual_tokens - estimated_tokens
    db.add(client)
    db.commit()
