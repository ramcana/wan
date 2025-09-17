from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str


class APIKeyCreate(BaseModel):
    name: str
    expires_in_days: Optional[int] = None


class APIKeyResponse(BaseModel):
    api_key: str
    key_id: str
    name: str
    expires_at: Optional[datetime]
    created_at: datetime