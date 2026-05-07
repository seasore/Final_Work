# -*- coding: utf-8 -*-
"""电力预测系统 - 后端主入口"""
import sys
import io
if sys.platform == 'win32' and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pymongo.errors import ServerSelectionTimeoutError

from config import settings
from db import init_db
from routes import auth, prediction, admin, messages, reports, units, holiday, weather, analytics

app = FastAPI(title="短期负荷预测管理系统", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(prediction.router)
app.include_router(weather.router)
app.include_router(analytics.router)
app.include_router(admin.router)
app.include_router(messages.router)
app.include_router(reports.router)
app.include_router(units.router)
app.include_router(holiday.router)


@app.on_event("startup")
def startup():
    try:
        init_db()
    except ServerSelectionTimeoutError as e:
        print(
            "\n========== MongoDB 未就绪 ==========\n"
            f"无法连接: {settings.MONGODB_URL}\n\n"
            "请先在本机启动 MongoDB（监听 27017），例如：\n"
            "  • 管理员 PowerShell:  net start MongoDB\n"
            "  • 或手动:  mongod --dbpath D:\\MongoData\\db\n"
            "  • 若用 Docker:  docker run -d -p 27017:27017 --name mongo mongo:7\n"
            "确认后再执行: python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000\n"
            "====================================\n",
            file=sys.stderr,
        )
        raise e


@app.get("/")
def root():
    return {"message": "电力负荷预测系统 API", "docs": "/docs"}


@app.get("/api/health")
def health():
    """无需登录：检查 API 进程与 MongoDB（浏览器可访问 /api/health）。"""
    try:
        from db import get_db

        get_db().command("ping")
        return {"ok": True, "api": True, "mongodb": True}
    except Exception as e:
        return {"ok": False, "api": True, "mongodb": False, "hint": "MongoDB 未启动或连接串错误", "error": str(e)[:300]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
