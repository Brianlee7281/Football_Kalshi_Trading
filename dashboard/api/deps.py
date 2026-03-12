# dashboard/api/deps.py
#
# FastAPI dependency providers shared across all route modules.

from __future__ import annotations

from typing import Annotated

import asyncpg
from fastapi import Depends, Request


async def get_pool(request: Request) -> asyncpg.Pool:
    pool: asyncpg.Pool = request.app.state.pool
    return pool


# Convenience alias — inject with: pool: Pool
Pool = Annotated[asyncpg.Pool, Depends(get_pool)]
