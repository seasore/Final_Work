# -*- coding: utf-8 -*-
"""节假日配置"""
from fastapi import APIRouter, Depends
from pymongo.database import Database
from bson import ObjectId
from pydantic import BaseModel

from db import get_db
from auth import get_current_admin

router = APIRouter(prefix="/api/holiday", tags=["holiday"])


class HolidayCreate(BaseModel):
    name: str
    start_date: str
    end_date: str
    type: str = "法定"


@router.get("/")
def list_holidays(db: Database = Depends(get_db)):
    items = list(db.holiday_config.find({}))
    return {"data": [{"id": str(u["_id"]), **{k: v for k, v in u.items() if k != "_id"}} for u in items]}


@router.post("/")
def create_holiday(data: HolidayCreate, db: Database = Depends(get_db), _=Depends(get_current_admin)):
    r = db.holiday_config.insert_one({"name": data.name, "start_date": data.start_date, "end_date": data.end_date, "type": data.type})
    return {"id": str(r.inserted_id)}
