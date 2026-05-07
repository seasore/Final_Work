# -*- coding: utf-8 -*-
"""汇报路由"""
from fastapi import APIRouter, Depends, HTTPException
from pymongo.database import Database
from pydantic import BaseModel
from bson import ObjectId

from db import get_db
from auth import get_current_user

router = APIRouter(prefix="/api/reports", tags=["reports"])

# 紧急程度：1=低 2=中 3=高 4=紧急
URGENCY_LEVELS = ["1", "2", "3", "4"]


class ReportCreate(BaseModel):
    content: str
    urgency: str  # 1-4


@router.post("/")
def create_report(
    data: ReportCreate,
    user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    """成员向上汇报，必须选择紧急程度"""
    if data.urgency not in URGENCY_LEVELS:
        raise HTTPException(status_code=400, detail="紧急程度必须为 1-4")
    doc = {
        "reporter_id": ObjectId(user["id"]) if user.get("id") else user["id"],
        "content": data.content,
        "urgency": data.urgency,
        "replied": False,
        "created_at": __import__("datetime").datetime.utcnow(),
    }
    r = db.reports.insert_one(doc)
    return {"id": str(r.inserted_id), "ok": True}


@router.get("/my")
def list_my_reports(user: dict = Depends(get_current_user), db: Database = Depends(get_db)):
    """我的汇报列表（与 create 一致：reporter_id 在库内为 ObjectId，查询须转换）"""
    uid = user.get("id")
    try:
        qid = ObjectId(uid) if uid else None
    except Exception:
        qid = uid
    if qid is None:
        return {"data": []}
    reports = list(db.reports.find({"reporter_id": qid}).sort("created_at", -1))
    out = []
    for r in reports:
        item = {
            "id": str(r["_id"]),
            "content": r.get("content"),
            "urgency": r.get("urgency"),
            "replied": r.get("replied", False),
            "reply_content": r.get("reply_content"),
            "created_at": r.get("created_at"),
        }
        rid = r.get("reporter_id")
        if isinstance(rid, ObjectId):
            item["reporter_id"] = str(rid)
        out.append(item)
    return {"data": out}
