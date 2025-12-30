"""
API Keys Endpoints
"""
from typing import List, Optional
from uuid import UUID
from datetime import datetime, timedelta
import secrets
import hashlib

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel

from app.core.database import get_db
from app.models.api import APIKey

router = APIRouter()

# Demo workspace ID
DEMO_WORKSPACE_ID = "00000000-0000-0000-0000-000000000001"


class APIKeyResponse(BaseModel):
    id: UUID
    name: str
    key_prefix: str
    permissions: List[str]
    rate_limit_per_minute: int
    last_used_at: Optional[datetime]
    expires_at: Optional[datetime]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class APIKeyCreate(BaseModel):
    name: str
    permissions: List[str] = ["*"]
    expires_in_days: Optional[int] = None  # None means never


class APIKeyCreateResponse(BaseModel):
    id: UUID
    name: str
    key: str  # Full key, only shown once
    key_prefix: str
    permissions: List[str]
    expires_at: Optional[datetime]
    created_at: datetime


def generate_api_key(prefix: str = "va_live_") -> tuple[str, str, str]:
    """Generate an API key and return (full_key, key_hash, key_prefix)."""
    random_part = secrets.token_urlsafe(32)
    full_key = f"{prefix}{random_part}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    key_prefix = full_key[:16]
    return full_key, key_hash, key_prefix


@router.get("", response_model=List[APIKeyResponse])
async def list_api_keys(
    db: AsyncSession = Depends(get_db),
):
    """List all API keys for the workspace."""
    query = (
        select(APIKey)
        .where(APIKey.workspace_id == DEMO_WORKSPACE_ID)
        .order_by(desc(APIKey.created_at))
    )
    result = await db.execute(query)
    return result.scalars().all()


@router.post("", response_model=APIKeyCreateResponse)
async def create_api_key(
    data: APIKeyCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new API key."""
    # Generate the key
    full_key, key_hash, key_prefix = generate_api_key()

    # Calculate expiration
    expires_at = None
    if data.expires_in_days:
        expires_at = datetime.now() + timedelta(days=data.expires_in_days)

    # Create the key
    api_key = APIKey(
        workspace_id=DEMO_WORKSPACE_ID,
        name=data.name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        permissions=data.permissions,
        expires_at=expires_at,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    return APIKeyCreateResponse(
        id=api_key.id,
        name=api_key.name,
        key=full_key,  # Only returned once
        key_prefix=key_prefix,
        permissions=api_key.permissions,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
    )


@router.get("/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get an API key by ID."""
    result = await db.execute(
        select(APIKey).where(APIKey.id == str(key_id))
    )
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    return api_key


@router.delete("/{key_id}")
async def revoke_api_key(
    key_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Revoke (delete) an API key."""
    result = await db.execute(
        select(APIKey).where(APIKey.id == str(key_id))
    )
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    await db.delete(api_key)
    await db.commit()
    return {"message": "API key revoked"}
