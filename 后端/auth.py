# -*- coding: utf-8 -*-
"""认证模块"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pymongo.database import Database

from bson import ObjectId
from config import settings
from db import get_db
from models.user import UserRole, UserResponse

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        return None


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Database = Depends(get_db)
) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效的认证凭据"
    )
    payload = decode_token(token)
    if payload is None:
        raise credentials_exception
    user_id: str = payload.get("sub")
    if not user_id:
        raise credentials_exception
    try:
        oid = ObjectId(user_id)
    except Exception:
        raise credentials_exception
    user = db.users.find_one({"_id": oid})
    if not user:
        raise credentials_exception
    user["id"] = str(user["_id"])
    if user.get("disabled"):
        raise HTTPException(status_code=403, detail="账号已被禁用")
    return user


async def get_current_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") not in (UserRole.GROUP_ADMIN, UserRole.TOTAL_ADMIN):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


async def get_current_total_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != UserRole.TOTAL_ADMIN:
        raise HTTPException(status_code=403, detail="需要总管理员权限")
    return user
