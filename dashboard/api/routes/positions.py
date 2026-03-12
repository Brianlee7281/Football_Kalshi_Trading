# dashboard/api/routes/positions.py
#
# REST endpoints:
#   GET /api/positions   → List[PositionItem]

from __future__ import annotations

from fastapi import APIRouter

from dashboard.api.deps import Pool
from dashboard.api.models import PositionItem
from dashboard.api.routes.matches import _row_to_position

router = APIRouter(prefix="/api", tags=["positions"])


@router.get("/positions", response_model=list[PositionItem])
async def list_positions(
    pool: Pool,
    status: str | None = "OPEN",
    is_paper: bool | None = None,
) -> list[PositionItem]:
    """All positions, filterable by status and paper/live mode.

    Default: status='OPEN'.  Pass status=None for all statuses.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, match_id, market_ticker, direction, entry_price, quantity,
                   status, is_paper, entry_time, exit_time, exit_price,
                   settlement_price, realized_pnl
            FROM positions
            WHERE ($1::text IS NULL OR status = $1)
              AND ($2::boolean IS NULL OR is_paper = $2)
            ORDER BY entry_time DESC
            LIMIT 500
            """,
            status,
            is_paper,
        )
    return [_row_to_position(r) for r in rows]
