"""
Phone Numbers Endpoints
"""
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from pydantic import BaseModel

from app.core.database import get_db
from app.models import PhoneNumber, Call

router = APIRouter()

# Demo workspace ID
DEMO_WORKSPACE_ID = "00000000-0000-0000-0000-000000000001"


class PhoneNumberResponse(BaseModel):
    id: UUID
    workspace_id: UUID
    agent_id: Optional[UUID]
    phone_number: str
    twilio_sid: Optional[str]
    country: str
    capabilities: dict
    monthly_cost_cents: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PhoneNumberStats(BaseModel):
    total_numbers: int
    active_numbers: int
    inbound_calls_this_month: int
    outbound_calls_this_month: int


class PhoneNumberCreate(BaseModel):
    phone_number: str
    country: str = "TR"
    agent_id: Optional[str] = None


class PhoneNumberUpdate(BaseModel):
    agent_id: Optional[str] = None
    is_active: Optional[bool] = None


@router.get("", response_model=List[PhoneNumberResponse])
async def list_phone_numbers(
    is_active: Optional[bool] = None,
    country: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List all phone numbers."""
    query = select(PhoneNumber).where(PhoneNumber.workspace_id == DEMO_WORKSPACE_ID)

    if is_active is not None:
        query = query.where(PhoneNumber.is_active == is_active)
    if country:
        query = query.where(PhoneNumber.country == country)

    query = query.order_by(desc(PhoneNumber.created_at))

    result = await db.execute(query)
    return result.scalars().all()


@router.get("/stats", response_model=PhoneNumberStats)
async def get_phone_number_stats(
    db: AsyncSession = Depends(get_db),
):
    """Get phone number statistics."""
    # Count phone numbers
    numbers_query = select(PhoneNumber).where(
        PhoneNumber.workspace_id == DEMO_WORKSPACE_ID
    )
    result = await db.execute(numbers_query)
    phone_numbers = result.scalars().all()

    total_numbers = len(phone_numbers)
    active_numbers = sum(1 for p in phone_numbers if p.is_active)

    # Count calls this month
    from datetime import datetime
    now = datetime.now()
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    calls_query = select(Call).where(
        Call.workspace_id == DEMO_WORKSPACE_ID,
        Call.started_at >= start_of_month
    )
    result = await db.execute(calls_query)
    calls = result.scalars().all()

    inbound_calls = sum(1 for c in calls if c.channel == "phone")
    outbound_calls = 0  # We don't track outbound yet

    return PhoneNumberStats(
        total_numbers=total_numbers,
        active_numbers=active_numbers,
        inbound_calls_this_month=inbound_calls,
        outbound_calls_this_month=outbound_calls,
    )


@router.get("/{phone_id}", response_model=PhoneNumberResponse)
async def get_phone_number(
    phone_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a phone number by ID."""
    result = await db.execute(
        select(PhoneNumber).where(PhoneNumber.id == str(phone_id))
    )
    phone = result.scalar_one_or_none()
    if not phone:
        raise HTTPException(status_code=404, detail="Phone number not found")
    return phone


@router.post("", response_model=PhoneNumberResponse)
async def create_phone_number(
    data: PhoneNumberCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new phone number."""
    phone = PhoneNumber(
        workspace_id=DEMO_WORKSPACE_ID,
        phone_number=data.phone_number,
        country=data.country,
        agent_id=data.agent_id,
    )
    db.add(phone)
    await db.commit()
    await db.refresh(phone)
    return phone


@router.patch("/{phone_id}", response_model=PhoneNumberResponse)
async def update_phone_number(
    phone_id: UUID,
    data: PhoneNumberUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a phone number."""
    result = await db.execute(
        select(PhoneNumber).where(PhoneNumber.id == str(phone_id))
    )
    phone = result.scalar_one_or_none()
    if not phone:
        raise HTTPException(status_code=404, detail="Phone number not found")

    if data.agent_id is not None:
        phone.agent_id = data.agent_id if data.agent_id else None
    if data.is_active is not None:
        phone.is_active = data.is_active

    await db.commit()
    await db.refresh(phone)
    return phone


@router.delete("/{phone_id}")
async def delete_phone_number(
    phone_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete (release) a phone number."""
    result = await db.execute(
        select(PhoneNumber).where(PhoneNumber.id == str(phone_id))
    )
    phone = result.scalar_one_or_none()
    if not phone:
        raise HTTPException(status_code=404, detail="Phone number not found")

    await db.delete(phone)
    await db.commit()
    return {"message": "Phone number released"}
