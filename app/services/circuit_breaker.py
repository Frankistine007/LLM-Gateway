from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app import models

# 3 in a row is enough to distinguish "this provider is actually down" from
# "one request happened to time out" without waiting through a long losing
# streak first.
FAILURE_THRESHOLD = 3

# How long to stop sending traffic to a tripped provider before testing it
# again. Short enough that a real recovery is noticed quickly; long enough
# that a still-down provider isn't re-tested on every request.
COOLDOWN_SECONDS = 30


class CircuitOpenError(Exception):
    """Raised instead of attempting a provider whose breaker is open."""

    def __init__(self, provider_name: str):
        super().__init__(f"{provider_name} circuit breaker is open")
        self.provider_name = provider_name


def _get_or_create(db: Session, provider_name: str) -> models.ProviderHealth:
    row = (
        db.query(models.ProviderHealth)
        .filter(models.ProviderHealth.provider == provider_name)
        .one_or_none()
    )
    if row is None:
        row = models.ProviderHealth(
            provider=provider_name, consecutive_failures=0, state="closed"
        )
        db.add(row)
        db.flush()
    return row


def is_open(db: Session, provider_name: str) -> bool:
    """True if this provider should be skipped entirely right now.

    Transitions open -> half_open once the cooldown elapses, and lets exactly
    one request through as a test (returns False for it) rather than fully
    resetting — a real recovery is confirmed by that one request succeeding,
    not assumed. Every DB write here goes through the same BEGIN IMMEDIATE
    locking as the rate limiter, so concurrent requests can't all read
    "cooldown just elapsed" and all become the test request at once.
    """
    row = _get_or_create(db, provider_name)

    if row.state != "open":
        db.commit()
        return False

    opened_at = row.opened_at
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=timezone.utc)

    elapsed = (datetime.now(timezone.utc) - opened_at).total_seconds()
    if elapsed < COOLDOWN_SECONDS:
        db.commit()
        return True

    row.state = "half_open"
    db.add(row)
    db.commit()
    return False


def record_success(db: Session, provider_name: str):
    row = _get_or_create(db, provider_name)
    row.consecutive_failures = 0
    row.state = "closed"
    row.opened_at = None
    db.add(row)
    db.commit()


def record_failure(db: Session, provider_name: str):
    """Only call this for retryable failures — a bad request (400/401) is the
    caller's fault, not evidence the provider is down, and shouldn't count.
    """
    row = _get_or_create(db, provider_name)
    row.consecutive_failures += 1

    # A half-open test request failing means the provider isn't actually
    # back yet: reopen immediately and restart the cooldown, rather than
    # waiting for the failure count to cross the threshold again.
    if row.state == "half_open" or row.consecutive_failures >= FAILURE_THRESHOLD:
        row.state = "open"
        row.opened_at = datetime.now(timezone.utc)

    db.add(row)
    db.commit()
