from datetime import datetime, timezone

from app import models

HEADER_NAMES = [
    "X-RateLimit-Limit-Requests",
    "X-RateLimit-Remaining-Requests",
    "X-RateLimit-Reset-Requests",
    "X-RateLimit-Limit-Tokens",
    "X-RateLimit-Remaining-Tokens",
    "X-RateLimit-Reset-Tokens",
]


def test_headers_present_on_successful_non_streaming(client, api_key, db_session):
    resp = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": api_key},
        json={
            "model": "llama-3.1-8b-instant",
            "stream": False,
            "messages": [{"role": "user", "content": "Say 'pong' and nothing else."}],
        },
    )

    assert resp.status_code == 200
    for name in HEADER_NAMES:
        assert name in resp.headers

    log_client = (
        db_session.query(models.Client).filter(models.Client.api_key == api_key).one()
    )
    assert resp.headers["X-RateLimit-Limit-Requests"] == str(log_client.rate_limit)
    # one request was just debited from a full bucket
    assert resp.headers["X-RateLimit-Remaining-Requests"] == str(
        log_client.rate_limit - 1
    )
    # tokens were debited, so the bucket is no longer full and needs time to reset
    assert int(resp.headers["X-RateLimit-Remaining-Tokens"]) < log_client.token_limit
    assert int(resp.headers["X-RateLimit-Reset-Tokens"]) > 0


def test_headers_present_on_streaming(client, api_key):
    resp = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": api_key},
        json={
            "model": "llama-3.1-8b-instant",
            "stream": True,
            "messages": [{"role": "user", "content": "Say 'pong' and nothing else."}],
        },
    )

    assert resp.status_code == 200
    for name in HEADER_NAMES:
        assert name in resp.headers


def test_headers_present_on_429_rejection(client, api_key, db_session):
    log_client = (
        db_session.query(models.Client).filter(models.Client.api_key == api_key).one()
    )
    log_client.bucket_requests = 0.0
    log_client.bucket_tokens = float(log_client.token_limit)
    log_client.bucket_updated_at = datetime.now(timezone.utc)
    db_session.add(log_client)
    db_session.commit()

    resp = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": api_key},
        json={
            "model": "llama-3.1-8b-instant",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert resp.status_code == 429
    for name in HEADER_NAMES:
        assert name in resp.headers

    assert resp.headers["X-RateLimit-Remaining-Requests"] == "0"
    assert int(resp.headers["X-RateLimit-Reset-Requests"]) > 0
    assert "Retry-After" in resp.headers
