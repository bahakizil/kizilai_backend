"""
Workspace Endpoints
"""
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.core.database import get_db
from app.models import Workspace, WorkspaceMember

router = APIRouter()

# Demo workspace ID
DEMO_WORKSPACE_ID = "00000000-0000-0000-0000-000000000001"


class WorkspaceResponse(BaseModel):
    id: UUID
    name: str
    slug: str
    logo_url: str | None
    plan: str
    timezone: str
    default_language: str

    class Config:
        from_attributes = True


class WorkspaceCreate(BaseModel):
    name: str
    slug: str
    timezone: str = "UTC"
    default_language: str = "tr"


class WorkspaceUpdate(BaseModel):
    name: str | None = None
    logo_url: str | None = None
    timezone: str | None = None
    default_language: str | None = None


@router.get("", response_model=List[WorkspaceResponse])
async def list_workspaces(db: AsyncSession = Depends(get_db)):
    """List all workspaces for current user."""
    result = await db.execute(select(Workspace))
    workspaces = result.scalars().all()
    return workspaces


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(workspace_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get workspace by ID."""
    result = await db.execute(select(Workspace).where(Workspace.id == str(workspace_id)))
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace


@router.post("", response_model=WorkspaceResponse)
async def create_workspace(
    data: WorkspaceCreate, db: AsyncSession = Depends(get_db)
):
    """Create new workspace."""
    workspace = Workspace(
        name=data.name,
        slug=data.slug,
        timezone=data.timezone,
        default_language=data.default_language,
    )
    db.add(workspace)
    await db.commit()
    await db.refresh(workspace)
    return workspace


@router.patch("/{workspace_id}", response_model=WorkspaceResponse)
async def update_workspace(
    workspace_id: UUID,
    data: WorkspaceUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update workspace."""
    result = await db.execute(select(Workspace).where(Workspace.id == str(workspace_id)))
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(workspace, field, value)

    await db.commit()
    await db.refresh(workspace)
    return workspace


@router.get("/{workspace_id}/members")
async def list_workspace_members(
    workspace_id: UUID, db: AsyncSession = Depends(get_db)
):
    """List workspace members."""
    result = await db.execute(
        select(WorkspaceMember).where(WorkspaceMember.workspace_id == str(workspace_id))
    )
    members = result.scalars().all()
    return members
