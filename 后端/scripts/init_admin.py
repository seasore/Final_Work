# -*- coding: utf-8 -*-
"""初始化管理员账号"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymongo import MongoClient
from config import settings
from auth import get_password_hash

client = MongoClient(settings.MONGODB_URL)
db = client[settings.MONGODB_DB]

# 创建总管理员
if not db.users.find_one({"username": "admin"}):
    db.users.insert_one({
        "username": "admin",
        "hashed_password": get_password_hash("admin123"),
        "role": "total_admin",
        "group_id": None,
        "disabled": False,
        "created_at": __import__("datetime").datetime.utcnow(),
    })
    print("已创建总管理员: admin / admin123")
else:
    print("总管理员已存在")

# 创建小组管理员示例
if not db.users.find_one({"username": "group1_admin"}):
    db.users.insert_one({
        "username": "group1_admin",
        "hashed_password": get_password_hash("group123"),
        "role": "group_admin",
        "group_id": "group1",
        "disabled": False,
        "created_at": __import__("datetime").datetime.utcnow(),
    })
    print("已创建小组管理员: group1_admin / group123 (组: group1)")
