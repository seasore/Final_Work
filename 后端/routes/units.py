# -*- coding: utf-8 -*-
"""单位管理"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pymongo.database import Database
from bson import ObjectId
from pydantic import BaseModel

from db import get_db
from auth import get_current_admin

router = APIRouter(prefix="/api/units", tags=["units"])


class UnitCreate(BaseModel):
    name: str
    code: str
    region: str = ""


@router.get("/")
def list_units(db: Database = Depends(get_db), _=Depends(get_current_admin)):
    units = list(db.units.find({}))
    return {"data": [{"id": str(u["_id"]), **{k: v for k, v in u.items() if k != "_id"}} for u in units]}


@router.post("/")
def create_unit(data: UnitCreate, db: Database = Depends(get_db), _=Depends(get_current_admin)):
    if db.units.find_one({"code": data.code}):
        raise HTTPException(400, "编号已存在")
    r = db.units.insert_one({"name": data.name, "code": data.code, "region": data.region})
    return {"id": str(r.inserted_id)}


class UnitUpdate(BaseModel):
    name: Optional[str] = None
    code: Optional[str] = None
    region: Optional[str] = None


@router.patch("/{uid}")
def update_unit(uid: str, data: UnitUpdate, db: Database = Depends(get_db), _=Depends(get_current_admin)):
    upd = {k: v for k, v in data.model_dump().items() if v is not None}
    if not upd:
        return {"ok": True}
    if "code" in upd and db.units.find_one({"code": upd["code"], "_id": {"$ne": ObjectId(uid)}}):
        raise HTTPException(400, "编号已存在")
    db.units.update_one({"_id": ObjectId(uid)}, {"$set": upd})
    return {"ok": True}


@router.delete("/{uid}")
def delete_unit(uid: str, db: Database = Depends(get_db), _=Depends(get_current_admin)):
    db.units.delete_one({"_id": ObjectId(uid)})
    return {"ok": True}
