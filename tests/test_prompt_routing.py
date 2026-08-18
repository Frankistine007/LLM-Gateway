from app.providers.registry import ROUTING_TIERS
from app.services.routing import classify_prompt


def test_classify_short_plain_prompt_as_simple():
    messages = [{"role": "user", "content": "What's the capital of France?"}]
    assert classify_prompt(messages) == "simple"


def test_classify_long_prompt_as_complex():
    messages = [{"role": "user", "content": "Explain this in detail. " * 60}]
    assert classify_prompt(messages) == "complex"


def test_classify_short_code_prompt_as_complex():
    messages = [{"role": "user", "content": "Fix this:\n```def f(x): return x+"}]
    assert classify_prompt(messages) == "complex"


def test_classify_looks_at_all_messages_combined():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "Explain this in detail. " * 60},
    ]
    assert classify_prompt(messages) == "complex"


def test_no_model_routes_to_a_tier(client, api_key):
    resp = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": api_key},
        json={
            "stream": False,
            "messages": [{"role": "user", "content": "Say 'pong' and nothing else."}],
        },
    )

    assert resp.status_code == 200
    assert resp.headers["X-Gateway-Routing"] == "auto:simple"
    assert resp.headers["X-Gateway-Model"] == ROUTING_TIERS["simple"]


def test_auto_model_routes_to_a_tier(client, api_key):
    resp = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": api_key},
        json={
            "model": "auto",
            "stream": False,
            "messages": [{"role": "user", "content": "Explain this in detail. " * 60}],
        },
    )

    assert resp.status_code == 200
    assert resp.headers["X-Gateway-Routing"] == "auto:complex"
    assert resp.headers["X-Gateway-Model"] == ROUTING_TIERS["complex"]


def test_explicit_model_is_always_honored(client, api_key):
    resp = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": api_key},
        json={
            "model": "openai/gpt-oss-120b",
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert resp.status_code == 200
    assert resp.headers["X-Gateway-Routing"] == "explicit"
    assert resp.headers["X-Gateway-Model"] == "openai/gpt-oss-120b"
