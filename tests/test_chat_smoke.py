from app import models


def test_non_streaming_chat_completion_happy_path(client, api_key, db_session):
    resp = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": api_key},
        json={
            "model": "openai/gpt-oss-20b",
            "stream": False,
            "messages": [{"role": "user", "content": "Say 'pong' and nothing else."}],
        },
    )

    assert resp.status_code == 200
    assert resp.text.strip() != ""

    log = (
        db_session.query(models.RequestLog)
        .order_by(models.RequestLog.id.desc())
        .first()
    )
    assert log is not None
    assert log.provider == "groq"
    assert log.model == "openai/gpt-oss-20b"
    assert log.response == resp.text
    assert log.tokens_used is not None and log.tokens_used > 0
    assert log.error_message is None
