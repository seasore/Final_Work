# -*- coding: utf-8 -*-
"""认证路由"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from db import get_db
from auth import get_password_hash, verify_password, create_access_token
from models.user import UserCreate, UserResponse, Token, UserRole

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Database = Depends(get_db)):
    """登录"""
    user = db.users.find_one({"username": form.username})
    if not user or not verify_password(form.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    if user.get("disabled"):
        raise HTTPException(status_code=403, detail="账号已被禁用")
    token = create_access_token(data={"sub": str(user["_id"])})
    return Token(
        access_token=token,
        user=UserResponse(
            id=str(user["_id"]),
            username=user["username"],
            role=user.get("role", UserRole.MEMBER),
            group_id=user.get("group_id"),
            disabled=user.get("disabled", False),
            created_at=user.get("created_at"),
        ),
    )


@router.post("/register", response_model=UserResponse)
def register(data: UserCreate, db: Database = Depends(get_db)):
    """注册（仅成员，管理员由总管理员创建）"""
    if db.users.find_one({"username": data.username}):
        raise HTTPException(status_code=400, detail="用户名已存在")
    gid = data.group_id
    if gid is not None and str(gid).strip() == "":
        gid = None
    doc = {
        "username": data.username,
        "hashed_password": get_password_hash(data.password),
        "role": UserRole.MEMBER.value,
        "group_id": gid,
        "disabled": False,
        "created_at": __import__("datetime").datetime.utcnow(),
    }
    try:
        r = db.users.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="用户名已存在")
    doc["_id"] = r.inserted_id
    return UserResponse(
        id=str(doc["_id"]),
        username=doc["username"],
        role=UserRole(doc["role"]),
        group_id=doc.get("group_id"),
        disabled=doc.get("disabled", False),
        created_at=doc["created_at"],
    )
