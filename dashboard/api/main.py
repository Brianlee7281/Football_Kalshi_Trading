"""MMPP Dashboard API — FastAPI application entry point.

Usage:
    uvicorn dashboard.api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.common.db import create_pool

app = FastAPI(title="MMPP Dashboard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────
from dashboard.api.routes.matches import router as matches_router
from dashboard.api.routes.positions import router as positions_router
from dashboard.api.routes.analytics import router as analytics_router
from dashboard.api.routes.system import router as system_router
from dashboard.api.routes.websocket import router as ws_router

app.include_router(matches_router)
app.include_router(positions_router)
app.include_router(analytics_router)
app.include_router(system_router)
app.include_router(ws_router)


@app.on_event("startup")
async def startup() -> None:
    db_url = os.environ.get(
        "DB_URL",
        "postgresql://postgres:postgres@localhost:5432/soccer_trading",
    )
    app.state.pool = await create_pool(db_url)


@app.on_event("shutdown")
async def shutdown() -> None:
    pool = getattr(app.state, "pool", None)
    if pool:
        await pool.close()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
