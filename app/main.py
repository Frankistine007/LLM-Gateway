from fastapi import FastAPI

from app.database import engine
from app import models
from app.routers import chat, clients

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="LLM Gateway")

app.include_router(clients.router)
app.include_router(chat.router)
