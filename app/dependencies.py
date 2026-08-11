from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app import models
from app.database import SessionLocal


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_client(
    x_api_key: str = Header(...),
    db: Session = Depends(get_db),
) -> models.Client:
    client = db.query(models.Client).filter(models.Client.api_key == x_api_key).first()
    if client is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return client
