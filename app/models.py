from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    api_key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    rate_limit = Column(Integer, nullable=False, server_default="10")
    token_limit = Column(Integer, nullable=False, server_default="100000")
    bucket_requests = Column(Float, nullable=True)
    bucket_tokens = Column(Float, nullable=True)
    bucket_updated_at = Column(DateTime(timezone=True), nullable=True)
    logs = relationship("RequestLog", back_populates="client")


class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    provider = Column(String, nullable=True)
    model = Column(String, nullable=True)
    prompt = Column(Text, nullable=True)
    response = Column(Text, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    latency = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    client = relationship("Client", back_populates="logs")


class ProviderHealth(Base):
    """Circuit breaker state, one row per provider (not per client — a
    provider outage affects every client, so this is shared, global state)."""

    __tablename__ = "provider_health"

    provider = Column(String, primary_key=True)
    consecutive_failures = Column(Integer, nullable=False, default=0)
    state = Column(String, nullable=False, default="closed")  # closed | open | half_open
    opened_at = Column(DateTime(timezone=True), nullable=True)
