from pydantic import BaseModel


class ClientCreate(BaseModel):
    name: str


class ClientOut(BaseModel):
    id: int
    name: str
    api_key: str
