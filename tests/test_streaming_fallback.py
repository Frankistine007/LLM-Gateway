from types import SimpleNamespace

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from app import models


def _chunk(text, usage_metadata=None):
    return SimpleNamespace(content=text, usage_metadata=usage_metadata)


def test_mid_stream_fallback_splices_in_backup_provider(
    monkeypatch, client, api_key, db_session
):
    def fake_primary_stream(self, messages, **kwargs):
        yield _chunk("Hello ", {"total_tokens": 5})
        yield _chunk("world", None)
        raise TimeoutError("simulated mid-stream timeout")

    def fake_fallback_stream(self, messages, **kwargs):
        yield _chunk("fallback reply", {"total_tokens": 3})

    monkeypatch.setattr(ChatGroq, "stream", fake_primary_stream)
    monkeypatch.setattr(ChatGoogleGenerativeAI, "stream", fake_fallback_stream)

    resp = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": api_key},
        json={
            "model": "openai/gpt-oss-20b",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert resp.status_code == 200
    body = resp.text

    # Text sent before the failure, and text from the fallback provider, both present.
    assert "Hello world" in body
    assert "fallback reply" in body

    # The splice notice only fires because tokens were already sent before failing.
    assert "'groq' failed mid-response" in body
    assert "switched to 'gemini'" in body
    assert "Text before and after this point may be inconsistent" in body

    log = (
        db_session.query(models.RequestLog)
        .order_by(models.RequestLog.id.desc())
        .first()
    )
    assert log is not None
    assert log.provider == "gemini"
    assert log.model == "gemini-flash-lite-latest"
    assert log.response == body
    assert "failed mid-stream" in log.error_message
