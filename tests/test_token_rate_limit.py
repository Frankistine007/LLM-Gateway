from datetime import datetime, timedelta, timezone

from app import models


def _get_client(db_session, api_key):
    return db_session.query(models.Client).filter(models.Client.api_key == api_key).one()


def test_request_bucket_exhausted_returns_429(client, api_key, db_session):
    log_client = _get_client(db_session, api_key)
    log_client.rate_limit = 5
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
    assert "Rate limit exceeded" in resp.json()["detail"]
    assert "Retry-After" in resp.headers


def test_token_bucket_insufficient_returns_429(client, api_key, db_session):
    log_client = _get_client(db_session, api_key)
    log_client.bucket_requests = float(log_client.rate_limit)
    log_client.bucket_tokens = 1.0  # far below any estimate for a real prompt
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
    assert "Token limit exceeded" in resp.json()["detail"]


def test_bucket_initializes_full_on_first_request(client, api_key, db_session):
    log_client = _get_client(db_session, api_key)
    assert log_client.bucket_updated_at is None

    # Every transaction now takes SQLite's write lock (BEGIN IMMEDIATE), so an
    # idle open transaction here would block the request under test.
    db_session.rollback()

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

    db_session.refresh(log_client)
    assert log_client.bucket_updated_at is not None
    # started full, minus 1 request and the estimated cost of this one
    assert log_client.bucket_requests == log_client.rate_limit - 1
    assert log_client.bucket_tokens < log_client.token_limit


def test_reconcile_refunds_overestimate_after_real_call(client, api_key, db_session):
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

    log_client = _get_client(db_session, api_key)
    log = (
        db_session.query(models.RequestLog)
        .filter(models.RequestLog.client_id == log_client.id)
        .order_by(models.RequestLog.id.desc())
        .first()
    )
    assert log.tokens_used is not None

    # Real replies are short; the 512-token default estimate should have been
    # refunded down to roughly what was actually used, not left fully debited.
    tokens_spent = log_client.token_limit - log_client.bucket_tokens
    assert tokens_spent < 512
    assert tokens_spent >= 0


def test_bucket_refills_over_time(client, api_key, db_session):
    log_client = _get_client(db_session, api_key)
    log_client.rate_limit = 10
    log_client.bucket_requests = 0.0
    log_client.bucket_tokens = float(log_client.token_limit)
    # Pretend the bucket was last touched 30s ago: at 10/60s refill rate,
    # that's +5 requests available now.
    log_client.bucket_updated_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    db_session.add(log_client)
    db_session.commit()

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
