# -*- coding: utf-8 -*-
"""用户模型"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class UserRole(str, Enum):
    """用户角色"""
    MEMBER = "member"           # 普通成员
    GROUP_ADMIN = "group_admin"  # 小组管理员
    TOTAL_ADMIN = "total_admin"  # 总管理员


class UserBase(BaseModel):
    username: str
    role: UserRole = UserRole.MEMBER
    group_id: Optional[str] = None  # 小组ID，group_admin 和 member 所属组
    disabled: bool = False


class UserCreate(UserBase):
    password: str


class UserInDB(UserBase):
    id: Optional[str] = None
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True


class UserResponse(UserBase):
    id: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
