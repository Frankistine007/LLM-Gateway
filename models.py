from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database import Base


class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    api_key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    rate_limit = Column(Integer, nullable=False, server_default="10")
    logs = relationship("RequestLog", back_populates="client")


class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    prompt = Column(Text, nullable=True)
    response = Column(Text, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    latency = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    client = relationship("Client", back_populates="logs")
