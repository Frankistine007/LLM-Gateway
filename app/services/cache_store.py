import time

CACHE_TTL_SECONDS = 300  # Keep an answer for 5 minutes.

# Each entry stores (answer_text, expiry_time).
_cache: dict[str, tuple[str, float]] = {}


def get_cached_response(key: str) -> str | None:
    entry = _cache.get(key)
    if entry is None:
        return None

    answer, expires_at = entry
    if time.monotonic() >= expires_at:
        del _cache[key]
        return None

    return answer


def set_cached_response(key: str, answer: str) -> None:
    expires_at = time.monotonic() + CACHE_TTL_SECONDS
    _cache[key] = (answer, expires_at)