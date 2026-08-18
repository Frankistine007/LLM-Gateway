from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from app import models
from app.services.circuit_breaker import COOLDOWN_SECONDS, FAILURE_THRESHOLD

PAYLOAD = {
    "model": "openai/gpt-oss-20b",
    "stream": False,
    "messages": [{"role": "user", "content": "hi"}],
}


class FakeProviderError(Exception):
    """Mimics a provider SDK exception carrying an HTTP-style status code,
    which is what is_retryable() actually inspects."""

    def __init__(self, status_code):
        super().__init__(f"boom {status_code}")
        self.status_code = status_code


def _ok_result(text="pong"):
    return SimpleNamespace(content=text, usage_metadata={"total_tokens": 5})


def _raise(status_code):
    def _invoke(self, messages, **kwargs):
        raise FakeProviderError(status_code)

    return _invoke


def _get_health(db_session, provider):
    return (
        db_session.query(models.ProviderHealth)
        .filter(models.ProviderHealth.provider == provider)
        .one()
    )


def test_breaker_trips_after_threshold_then_skips_primary(
    monkeypatch, client, api_key, db_session
):
    call_count = {"groq": 0}
    raise_503 = _raise(503)

    def fake_groq_invoke(self, messages, **kwargs):
        call_count["groq"] += 1
        return raise_503(self, messages, **kwargs)

    monkeypatch.setattr(ChatGroq, "invoke", fake_groq_invoke)
    monkeypatch.setattr(
        ChatGoogleGenerativeAI, "invoke", lambda self, messages, **kwargs: _ok_result()
    )

    for _ in range(FAILURE_THRESHOLD):
        resp = client.post(
            "/v1/chat/completions", headers={"X-API-Key": api_key}, json=PAYLOAD
        )
        assert resp.status_code == 200  # fallback covered it each time

    assert call_count["groq"] == FAILURE_THRESHOLD
    assert _get_health(db_session, "groq").state == "open"
    # Reading via db_session took SQLite's write lock (BEGIN IMMEDIATE); release
    # it before the next request, or that request blocks on this idle session.
    db_session.rollback()

    # Breaker is now open: the next request should skip Groq entirely and go
    # straight to the fallback, without incrementing the primary call count.
    resp = client.post(
        "/v1/chat/completions", headers={"X-API-Key": api_key}, json=PAYLOAD
    )
    assert resp.status_code == 200
    assert call_count["groq"] == FAILURE_THRESHOLD


def test_breaker_stays_closed_on_success(monkeypatch, client, api_key, db_session):
    monkeypatch.setattr(
        ChatGroq, "invoke", lambda self, messages, **kwargs: _ok_result()
    )

    resp = client.post(
        "/v1/chat/completions", headers={"X-API-Key": api_key}, json=PAYLOAD
    )
    assert resp.status_code == 200

    health = _get_health(db_session, "groq")
    assert health.state == "closed"
    assert health.consecutive_failures == 0


def test_non_retryable_failure_does_not_count_toward_trip(
    monkeypatch, client, api_key, db_session
):
    monkeypatch.setattr(ChatGroq, "invoke", _raise(400))

    resp = client.post(
        "/v1/chat/completions", headers={"X-API-Key": api_key}, json=PAYLOAD
    )
    assert resp.status_code == 502  # terminal error, no fallback attempted

    health = _get_health(db_session, "groq")
    assert health.consecutive_failures == 0
    assert health.state == "closed"


def test_half_open_success_closes_the_circuit(monkeypatch, client, api_key, db_session):
    # Trip the breaker directly rather than via real failing requests, and
    # backdate it past the cooldown so the next request is the half-open test.
    db_session.add(
        models.ProviderHealth(
            provider="groq",
            consecutive_failures=FAILURE_THRESHOLD,
            state="open",
            opened_at=datetime.now(timezone.utc)
            - timedelta(seconds=COOLDOWN_SECONDS + 5),
        )
    )
    db_session.commit()

    monkeypatch.setattr(
        ChatGroq, "invoke", lambda self, messages, **kwargs: _ok_result()
    )

    resp = client.post(
        "/v1/chat/completions", headers={"X-API-Key": api_key}, json=PAYLOAD
    )
    assert resp.status_code == 200

    health = _get_health(db_session, "groq")
    assert health.state == "closed"
    assert health.consecutive_failures == 0


def test_half_open_failure_reopens_immediately(monkeypatch, client, api_key, db_session):
    db_session.add(
        models.ProviderHealth(
            provider="groq",
            consecutive_failures=FAILURE_THRESHOLD,
            state="open",
            opened_at=datetime.now(timezone.utc)
            - timedelta(seconds=COOLDOWN_SECONDS + 5),
        )
    )
    db_session.commit()

    monkeypatch.setattr(ChatGroq, "invoke", _raise(503))
    monkeypatch.setattr(
        ChatGoogleGenerativeAI, "invoke", lambda self, messages, **kwargs: _ok_result()
    )

    resp = client.post(
        "/v1/chat/completions", headers={"X-API-Key": api_key}, json=PAYLOAD
    )
    assert resp.status_code == 200  # fallback still covers the caller

    health = _get_health(db_session, "groq")
    assert health.state == "open"
