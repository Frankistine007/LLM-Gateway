import json
import os
import secrets
import time

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

import models
from database import SessionLocal, engine

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

models.Base.metadata.create_all(bind=engine)

app = FastAPI()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
    x_api_key: str = Header(...), #  FastAPI's Header(...) automatically converts 
    #underscores in the parameter name to hyphens when matching the actual HTTP header.
    db: Session = Depends(get_db),
):
    client = db.query(models.Client).filter(models.Client.api_key == x_api_key).first()
    if client is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    body = await request.json()

    log = models.RequestLog(
        client_id=client.id,
        prompt=json.dumps(body.get("messages")),
    )

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            groq_response = await http_client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json=body,
            )
        groq_response.raise_for_status()
        data = groq_response.json()

        log.response = json.dumps(data.get("choices"))
        log.tokens_used = data.get("usage", {}).get("total_tokens")

        return data

    except httpx.HTTPStatusError as exc:
        try:
            groq_error = exc.response.json().get("error", {}).get("message", exc.response.text)
        except ValueError:
            groq_error = exc.response.text

        log.error_message = f"Groq {exc.response.status_code}: {groq_error}"
        raise HTTPException(status_code=exc.response.status_code, detail=groq_error) from exc

    except httpx.TimeoutException as exc:
        log.error_message = "Groq request timed out"
        raise HTTPException(status_code=504, detail="Groq request timed out") from exc

    except httpx.RequestError as exc:
        log.error_message = f"Network error calling Groq: {exc}"
        raise HTTPException(status_code=502, detail="Could not reach Groq") from exc

    finally:
        log.latency = time.perf_counter() - start
        db.add(log)
        db.commit()
