# -*- coding: utf-8 -*-
"""消息路由（成员端）"""
from fastapi import APIRouter, Depends
from pymongo.database import Database

from db import get_db
from auth import get_current_user

router = APIRouter(prefix="/api/messages", tags=["messages"])


@router.get("/")
def list_my_messages(user: dict = Depends(get_current_user), db: Database = Depends(get_db)):
    """获取当前用户收到的消息"""
    msgs = list(db.messages.find({"receiver_id": user["id"]}).sort("created_at", -1).limit(50))
    for m in msgs:
        m["id"] = str(m["_id"])
        sender = db.users.find_one({"_id": m["sender_id"]})
        m["sender_name"] = sender["username"] if sender else ""
    return {"data": msgs}


@router.patch("/{msg_id}/read")
def mark_read(msg_id: str, user: dict = Depends(get_current_user), db: Database = Depends(get_db)):
    """标记消息已读"""
    db.messages.update_one(
        {"_id": msg_id, "receiver_id": user["id"]},
        {"$set": {"read": True}}
    )
    return {"ok": True}
