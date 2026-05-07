# -*- coding: utf-8 -*-
"""管理员路由：用户管理、日志、消息、汇报"""
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pymongo.database import Database
from bson import ObjectId
from pydantic import BaseModel

from db import get_db
from auth import get_current_user, get_current_admin, get_current_total_admin, get_password_hash
from models.user import UserRole, UserCreate, UserResponse

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _user_response(doc: dict) -> dict:
    return {
        "id": str(doc["_id"]),
        "username": doc["username"],
        "role": doc.get("role", UserRole.MEMBER),
        "group_id": doc.get("group_id"),
        "disabled": doc.get("disabled", False),
        "created_at": doc.get("created_at"),
    }


# ---------- 用户管理 ----------
@router.get("/users", response_model=List[dict])
def list_users(
    admin: dict = Depends(get_current_admin),
    db: Database = Depends(get_db),
    role: Optional[str] = Query(None),
    group_id: Optional[str] = Query(None),
):
    """列出用户。总管理员看全部，小组管理员只看本组"""
    q = {}
    if admin["role"] == UserRole.GROUP_ADMIN:
        if not admin.get("group_id"):
            return []
        q["group_id"] = admin["group_id"]
    elif role:
        q["role"] = role
    if group_id:
        q["group_id"] = group_id
    users = list(db.users.find(q))
    return [_user_response(u) for u in users]


@router.post("/users")
def create_user(
    data: UserCreate,
    admin: dict = Depends(get_current_total_admin),
    db: Database = Depends(get_db),
):
    """创建用户（总管理员）"""
    if db.users.find_one({"username": data.username}):
        raise HTTPException(status_code=400, detail="用户名已存在")
    doc = {
        "username": data.username,
        "hashed_password": get_password_hash(data.password),
        "role": data.role,
        "group_id": data.group_id,
        "disabled": False,
        "created_at": datetime.utcnow(),
    }
    r = db.users.insert_one(doc)
    return {"id": str(r.inserted_id), "username": data.username}


class DisableBody(BaseModel):
    disabled: bool


class UpdateRoleBody(BaseModel):
    role: str


@router.patch("/users/{user_id}/role")
def update_user_role(
    user_id: str,
    body: UpdateRoleBody,
    admin: dict = Depends(get_current_total_admin),
    db: Database = Depends(get_db),
):
    """任命/撤销小组管理员（仅总管理员）"""
    try:
        oid = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="无效的用户ID")
    if body.role not in (UserRole.MEMBER, UserRole.GROUP_ADMIN):
        raise HTTPException(status_code=400, detail="角色只能是 member 或 group_admin")
    user = db.users.find_one({"_id": oid})
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    if user.get("role") == UserRole.TOTAL_ADMIN:
        raise HTTPException(status_code=403, detail="无法修改总管理员角色")
    db.users.update_one({"_id": oid}, {"$set": {"role": body.role}})
    return {"ok": True}


@router.patch("/users/{user_id}/disable")
def disable_user(
    user_id: str,
    body: DisableBody,
    admin: dict = Depends(get_current_admin),
    db: Database = Depends(get_db),
):
    """禁用/启用用户"""
    try:
        oid = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="无效的用户ID")
    user = db.users.find_one({"_id": oid})
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    if admin["role"] == UserRole.GROUP_ADMIN:
        if user.get("group_id") != admin.get("group_id"):
            raise HTTPException(status_code=403, detail="只能管理本组成员")
        if user.get("role") == UserRole.TOTAL_ADMIN:
            raise HTTPException(status_code=403, detail="无法操作总管理员")
    db.users.update_one({"_id": oid}, {"$set": {"disabled": body.disabled}})
    return {"ok": True}


# ---------- 消息 ----------
class SendMessageBody(BaseModel):
    receiver_ids: List[str]
    content: str


@router.post("/messages")
def send_message(
    body: SendMessageBody,
    admin: dict = Depends(get_current_admin),
    db: Database = Depends(get_db),
):
    """发送消息。总管理员可发全体，小组管理员发本组"""
    receiver_ids = body.receiver_ids
    content = body.content
    if admin["role"] == UserRole.TOTAL_ADMIN and "all" in receiver_ids:
        receiver_ids = [str(u["_id"]) for u in db.users.find({"disabled": {"$ne": True}})]
    elif admin["role"] == UserRole.GROUP_ADMIN and "all" in receiver_ids:
        receiver_ids = [str(u["_id"]) for u in db.users.find({"group_id": admin.get("group_id")})]
    elif admin["role"] == UserRole.GROUP_ADMIN:
        allowed = [str(u["_id"]) for u in db.users.find({"group_id": admin.get("group_id")})]
        receiver_ids = [x for x in receiver_ids if x in allowed]
    for rid in receiver_ids:
        db.messages.insert_one({
            "sender_id": admin["id"],
            "receiver_id": rid,
            "content": content,
            "created_at": datetime.utcnow(),
            "read": False,
        })
    return {"sent": len(receiver_ids)}


# ---------- 汇报 ----------
@router.get("/reports/unread")
def list_unread_reports(
    admin: dict = Depends(get_current_admin),
    db: Database = Depends(get_db),
    urgency: Optional[str] = Query(None),
):
    """未读汇报列表，按紧急程度优先、时间排序"""
    q = {"replied": {"$ne": True}}
    if urgency:
        q["urgency"] = urgency
    if admin["role"] == UserRole.GROUP_ADMIN:
        group_members = [u["_id"] for u in db.users.find({"group_id": admin.get("group_id")})]
        q["reporter_id"] = {"$in": group_members}
    reports = list(db.reports.find(q).sort([("urgency", -1), ("created_at", -1)]))
    for r in reports:
        r["id"] = str(r["_id"])
        try:
            u = db.users.find_one({"_id": ObjectId(r["reporter_id"])}) if isinstance(r.get("reporter_id"), str) else db.users.find_one({"_id": r["reporter_id"]})
        except Exception:
            u = db.users.find_one({"_id": r["reporter_id"]})
        r["reporter_name"] = u["username"] if u else ""
    return {"data": reports}


@router.get("/reports/read")
def list_read_reports(
    admin: dict = Depends(get_current_admin),
    db: Database = Depends(get_db),
    urgency: Optional[str] = Query(None),
):
    """已读汇报列表"""
    q = {"replied": True}
    if urgency:
        q["urgency"] = urgency
    if admin["role"] == UserRole.GROUP_ADMIN:
        group_members = [u["_id"] for u in db.users.find({"group_id": admin.get("group_id")})]
        q["reporter_id"] = {"$in": group_members}
    reports = list(db.reports.find(q).sort([("urgency", -1), ("created_at", -1)]))
    for r in reports:
        r["id"] = str(r["_id"])
        try:
            u = db.users.find_one({"_id": ObjectId(r["reporter_id"])}) if isinstance(r.get("reporter_id"), str) else db.users.find_one({"_id": r["reporter_id"]})
        except Exception:
            u = db.users.find_one({"_id": r["reporter_id"]})
        r["reporter_name"] = u["username"] if u else ""
    return {"data": reports}


@router.post("/reports/{report_id}/reply")
def reply_report(
    report_id: str,
    reply_content: str,
    admin: dict = Depends(get_current_admin),
    db: Database = Depends(get_db),
):
    """回复汇报，标记为已读"""
    try:
        oid = ObjectId(report_id)
    except Exception:
        raise HTTPException(status_code=400, detail="无效的汇报ID")
    r = db.reports.find_one({"_id": oid})
    if not r:
        raise HTTPException(status_code=404, detail="汇报不存在")
    if admin["role"] == UserRole.GROUP_ADMIN:
        group_member_ids = [u["_id"] for u in db.users.find({"group_id": admin.get("group_id")})]
        if r.get("reporter_id") not in group_member_ids:
            raise HTTPException(status_code=403, detail="只能回复本组汇报")
    db.reports.update_one(
        {"_id": oid},
        {"$set": {"replied": True, "reply_content": reply_content, "reply_at": datetime.utcnow(), "replier_id": admin["id"]}}
    )
    return {"ok": True}


# ---------- 日志 ----------
@router.get("/logs")
def list_logs(
    admin: dict = Depends(get_current_admin),
    db: Database = Depends(get_db),
    limit: int = Query(100, le=500),
):
    """查看操作日志"""
    q = {}
    if admin["role"] == UserRole.GROUP_ADMIN:
        group_members = [u["_id"] for u in db.users.find({"group_id": admin.get("group_id")})]
        q["user_id"] = {"$in": group_members}
    logs = list(db.operation_logs.find(q).sort("created_at", -1).limit(limit))
    for log in logs:
        log["id"] = str(log["_id"])
    return {"data": logs}


def log_operation(db: Database, user_id: str, action: str, detail: str = ""):
    """记录操作日志"""
    db.operation_logs.insert_one({
        "user_id": user_id,
        "action": action,
        "detail": detail,
        "created_at": datetime.utcnow(),
    })
