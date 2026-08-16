import threading
from datetime import datetime, timezone

from fastapi import HTTPException

from app import models
from app.services.rate_limit import reserve_capacity

THREADS = 8


def test_concurrent_reserves_cannot_overspend_one_slot(
    api_key, db_session, session_factory
):
    """Regression test for a read-modify-write race.

    Before BEGIN IMMEDIATE, all 8 threads read the same pre-debit bucket level,
    all passed the check, and then overwrote each other's debits — a full
    bypass of the limit, not an occasional double-spend.
    """
    log_client = (
        db_session.query(models.Client).filter(models.Client.api_key == api_key).one()
    )
    log_client.bucket_requests = 1.0  # capacity for exactly one request
    log_client.bucket_tokens = float(log_client.token_limit)
    log_client.bucket_updated_at = datetime.now(timezone.utc)
    client_id = log_client.id
    db_session.add(log_client)
    db_session.commit()
    # Release the write lock before the threads contend for it.
    db_session.rollback()

    results = []
    results_lock = threading.Lock()
    # Sync up before opening sessions: with BEGIN IMMEDIATE a query takes the
    # write lock, so waiting at a barrier mid-transaction would deadlock.
    barrier = threading.Barrier(THREADS)

    def attempt():
        barrier.wait()
        db = session_factory()
        try:
            client = db.query(models.Client).filter(models.Client.id == client_id).one()
            reserve_capacity(client, db, estimated_tokens=10)
            outcome = "allowed"
        except HTTPException:
            outcome = "rejected"
        finally:
            db.close()
        with results_lock:
            results.append(outcome)

    threads = [threading.Thread(target=attempt) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(results) == THREADS, "a thread deadlocked or died"
    assert results.count("allowed") == 1
    assert results.count("rejected") == THREADS - 1
