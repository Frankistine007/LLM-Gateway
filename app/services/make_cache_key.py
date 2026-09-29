import hashlib
import json

def make_cache_key(client_id, model, messages, temperature=None, max_tokens=None):
    request_details = {
        "client_id": client_id,
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Sort keys so the same details always produce the same JSON string.
    stable_text = json.dumps(
        request_details,
        sort_keys=True,
        separators=(",", ":"),
    )

    full_hash = hashlib.sha256(stable_text.encode("utf-8")).hexdigest()

    return full_hash  # Recommended: 64 characters
    # return full_hash[:16]  # Shorter 16-character version   