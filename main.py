import json
import os
import secrets
import time
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from langchain_groq import ChatGroq
from pydantic import BaseModel
from sqlalchemy.orm import Session

import models
from database import SessionLocal, engine

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

models.Base.metadata.create_all(bind=engine)

app = FastAPI()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


RATE_LIMIT_WINDOW_SECONDS = 60


def check_rate_limit(client: models.Client, db: Session):
    window_start = datetime.now(timezone.utc) - timedelta(
        seconds=RATE_LIMIT_WINDOW_SECONDS
    )

    recent_requests = (
        db.query(models.RequestLog)
        .filter(models.RequestLog.client_id == client.id)
        .filter(models.RequestLog.timestamp >= window_start)
        .count()
    )

    if recent_requests >= client.rate_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {client.rate_limit} requests per "
            f"{RATE_LIMIT_WINDOW_SECONDS}s",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
        )


class ClientCreate(BaseModel):
    name: str


@app.post("/clients")
def create_client(
    payload: ClientCreate,
    x_admin_key: str = Header(...),
    db: Session = Depends(get_db),
):
    if not secrets.compare_digest(x_admin_key, ADMIN_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid admin key")

    api_key = secrets.token_hex(16)
    client = models.Client(name=payload.name, api_key=api_key)
    db.add(client)
    db.commit()
    db.refresh(client)
    return {"id": client.id, "name": client.name, "api_key": client.api_key}


@app.post("/v1/chat/completions")
async def get_request_body(
    request: Request,
    x_api_key: str = Header(...),  # FastAPI's Header(...) automatically converts underscores in the parameter name to hyphens when matching the actual HTTP header.
    db: Session = Depends(get_db),
):
    client = db.query(models.Client).filter(models.Client.api_key == x_api_key).first()
    if client is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    check_rate_limit(client, db)

    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "llama-3.1-8b-instant")
    stream = body.get("stream", True)

    log = models.RequestLog(
        client_id=client.id,
        prompt=json.dumps(messages),
    )

    llm = ChatGroq(model=model, api_key=GROQ_API_KEY)
    lc_messages = [(m["role"], m["content"]) for m in messages]

    if not stream:
        start = time.perf_counter()
        try:
            result = llm.invoke(lc_messages)
            log.response = result.content
            return PlainTextResponse(result.content)
        except Exception as exc:
            log.error_message = str(exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            log.latency = time.perf_counter() - start
            db.add(log)
            db.commit()

    def token_stream():
        start = time.perf_counter()
        full_response = ""
        try:
            for chunk in llm.stream(lc_messages):
                full_response += chunk.content
                yield chunk.content
        except Exception as exc:
            log.error_message = str(exc)
        finally:
            log.response = full_response
            log.latency = time.perf_counter() - start
            db.add(log)
            db.commit()

    return StreamingResponse(token_stream(), media_type="text/plain")
