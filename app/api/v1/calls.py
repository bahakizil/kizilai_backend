"""
Call Endpoints
"""
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel

from app.core.database import get_db
from app.models import Call, Message

router = APIRouter()

# Demo workspace ID
DEMO_WORKSPACE_ID = "00000000-0000-0000-0000-000000000001"


class CallResponse(BaseModel):
    id: UUID
    agent_id: Optional[UUID]
    channel: str
    caller_id: Optional[str]
    status: str
    duration_seconds: Optional[int]
    message_count: int
    sentiment: Optional[str]
    cost_cents: int
    started_at: datetime
    ended_at: Optional[datetime]

    class Config:
        from_attributes = True


class CallDetailResponse(CallResponse):
    caller_phone: Optional[str]
    caller_name: Optional[str]
    end_reason: Optional[str]
    stt_seconds: float
    llm_tokens: int
    tts_characters: int
    avg_ttfa_ms: Optional[float]
    avg_llm_ms: Optional[float]
    avg_tts_ms: Optional[float]
    summary: Optional[str]
    user_rating: Optional[int]


class MessageResponse(BaseModel):
    id: UUID
    role: str
    content: Optional[str]
    stt_ms: Optional[float]
    llm_ms: Optional[float]
    tts_ms: Optional[float]
    ttfa_ms: Optional[float]
    function_name: Optional[str]
    function_result: Optional[dict]
    was_interrupted: bool
    created_at: datetime

    class Config:
        from_attributes = True


class CallStats(BaseModel):
    total_calls: int
    total_minutes: float
    avg_duration: float
    success_rate: float
    total_cost_cents: int


class DailyStats(BaseModel):
    date: str
    calls: int
    minutes: float


class UsageStatsResponse(BaseModel):
    daily: List[DailyStats]


@router.get("", response_model=List[CallResponse])
async def list_calls(
    agent_id: Optional[UUID] = None,
    status: Optional[str] = None,
    channel: Optional[str] = None,
    limit: int = Query(default=50, le=100),
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List calls with filters."""
    query = select(Call).where(Call.workspace_id == DEMO_WORKSPACE_ID)

    if agent_id:
        query = query.where(Call.agent_id == str(agent_id))
    if status:
        query = query.where(Call.status == status)
    if channel:
        query = query.where(Call.channel == channel)

    query = query.order_by(desc(Call.started_at)).offset(offset).limit(limit)

    result = await db.execute(query)
    calls = result.scalars().all()
    return calls


@router.get("/stats", response_model=CallStats)
async def get_call_stats(
    agent_id: Optional[UUID] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get call statistics."""
    query = select(Call).where(Call.workspace_id == DEMO_WORKSPACE_ID)
    if agent_id:
        query = query.where(Call.agent_id == str(agent_id))

    result = await db.execute(query)
    calls = result.scalars().all()

    total_calls = len(calls)
    total_seconds = sum(c.duration_seconds or 0 for c in calls)
    completed_calls = sum(1 for c in calls if c.status == "completed")

    return CallStats(
        total_calls=total_calls,
        total_minutes=total_seconds / 60,
        avg_duration=total_seconds / total_calls if total_calls > 0 else 0,
        success_rate=(completed_calls / total_calls * 100) if total_calls > 0 else 0,
        total_cost_cents=sum(c.cost_cents for c in calls),
    )


@router.get("/usage", response_model=UsageStatsResponse)
async def get_usage_stats(
    period: str = Query(default="week", regex="^(week|month|year)$"),
    db: AsyncSession = Depends(get_db),
):
    """Get daily usage statistics for charts."""
    from datetime import timedelta

    now = datetime.now()
    if period == "week":
        start_date = now - timedelta(days=7)
    elif period == "month":
        start_date = now - timedelta(days=30)
    else:  # year
        start_date = now - timedelta(days=365)

    query = select(Call).where(
        Call.workspace_id == DEMO_WORKSPACE_ID,
        Call.started_at >= start_date
    )
    result = await db.execute(query)
    calls = result.scalars().all()

    # Group by date
    daily_data: dict = {}
    for call in calls:
        date_key = call.started_at.strftime("%Y-%m-%d")
        if date_key not in daily_data:
            daily_data[date_key] = {"calls": 0, "minutes": 0.0}
        daily_data[date_key]["calls"] += 1
        daily_data[date_key]["minutes"] += (call.duration_seconds or 0) / 60

    # Fill in missing dates
    daily_stats = []
    current = start_date
    while current <= now:
        date_key = current.strftime("%Y-%m-%d")
        data = daily_data.get(date_key, {"calls": 0, "minutes": 0.0})
        daily_stats.append(DailyStats(
            date=date_key,
            calls=data["calls"],
            minutes=round(data["minutes"], 1)
        ))
        current += timedelta(days=1)

    return UsageStatsResponse(daily=daily_stats)


@router.get("/{call_id}", response_model=CallDetailResponse)
async def get_call(call_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get call details."""
    result = await db.execute(select(Call).where(Call.id == str(call_id)))
    call = result.scalar_one_or_none()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    return call


@router.get("/{call_id}/messages", response_model=List[MessageResponse])
async def get_call_messages(call_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get messages for a call."""
    result = await db.execute(
        select(Message)
        .where(Message.call_id == str(call_id))
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()
    return messages


@router.post("/{call_id}/rate")
async def rate_call(
    call_id: UUID,
    rating: int = Query(..., ge=1, le=5),
    db: AsyncSession = Depends(get_db),
):
    """Rate a call (1-5 stars)."""
    result = await db.execute(select(Call).where(Call.id == str(call_id)))
    call = result.scalar_one_or_none()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    call.user_rating = rating
    await db.commit()

    return {"message": "Rating saved", "rating": rating}
