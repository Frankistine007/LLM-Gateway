import secrets

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app import models
from app.config import settings
from app.dependencies import get_db
from app.schemas import ClientCreate, ClientOut

router = APIRouter(tags=["clients"])


@router.post("/clients", response_model=ClientOut)
def create_client(
    payload: ClientCreate,
    x_admin_key: str = Header(...),
    db: Session = Depends(get_db),
):
    if not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")

    api_key = secrets.token_hex(16)
    client = models.Client(name=payload.name, api_key=api_key)
    db.add(client)
    db.commit()
    db.refresh(client)
    return {"id": client.id, "name": client.name, "api_key": client.api_key}
