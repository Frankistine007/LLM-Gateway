# Status codes worth retrying elsewhere. A 400 or 401 means the request itself
# is bad, so retrying against another provider just wastes a second call.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
RETRYABLE_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "DEADLINE_EXCEEDED",
    "INTERNAL",
    "rate limit",
    "timeout",
    "overloaded",
)


def is_retryable(exc: Exception) -> bool:
    """Whether a provider failure justifies falling back to another provider."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        return status in RETRYABLE_STATUS

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    message = str(exc)
    return any(marker.lower() in message.lower() for marker in RETRYABLE_MARKERS)
