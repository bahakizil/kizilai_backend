"""
Authentication Utilities
JWT validation and user extraction for Supabase Auth
"""
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.database import get_db


# Security scheme
security = HTTPBearer(auto_error=False)


class AuthUser(BaseModel):
    """Authenticated user from JWT token."""
    id: UUID
    email: Optional[str] = None
    role: str = "authenticated"
    aud: str = "authenticated"
    exp: datetime
    iat: datetime

    class Config:
        from_attributes = True


class WorkspaceContext(BaseModel):
    """Current workspace context for the request."""
    workspace_id: UUID
    user_id: UUID
    role: str  # owner, admin, member


def decode_jwt(token: str) -> dict:
    """
    Decode and validate a Supabase JWT token.
    """
    try:
        payload = jwt.decode(
            token,
            settings.supabase.jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> AuthUser:
    """
    Get current authenticated user from JWT token.
    Use as a dependency in protected routes.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_jwt(token)

    return AuthUser(
        id=UUID(payload["sub"]),
        email=payload.get("email"),
        role=payload.get("role", "authenticated"),
        aud=payload.get("aud", "authenticated"),
        exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
        iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
    )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[AuthUser]:
    """
    Get current user if authenticated, otherwise return None.
    Use for routes that work both authenticated and unauthenticated.
    """
    if credentials is None:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


async def get_workspace_context(
    workspace_id: UUID,
    current_user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkspaceContext:
    """
    Get workspace context and verify user has access.
    Use as a dependency for workspace-scoped routes.
    """
    from app.models.workspace import WorkspaceMember

    result = await db.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == str(workspace_id),
            WorkspaceMember.user_id == str(current_user.id),
        )
    )
    member = result.scalar_one_or_none()

    if not member:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this workspace",
        )

    return WorkspaceContext(
        workspace_id=workspace_id,
        user_id=current_user.id,
        role=member.role,
    )


def require_role(*roles: str):
    """
    Dependency factory to require specific workspace roles.
    Usage: Depends(require_role("owner", "admin"))
    """
    async def check_role(context: WorkspaceContext = Depends(get_workspace_context)):
        if context.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires one of these roles: {', '.join(roles)}",
            )
        return context
    return check_role


# Convenience dependencies
require_owner = require_role("owner")
require_admin = require_role("owner", "admin")
require_member = require_role("owner", "admin", "member")


# Demo mode support (for development without auth)
DEMO_USER_ID = "00000000-0000-0000-0000-000000000001"
DEMO_WORKSPACE_ID = "00000000-0000-0000-0000-000000000001"


async def get_current_user_or_demo(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> AuthUser:
    """
    Get current user or return demo user in development mode.
    """
    if credentials:
        return await get_current_user(credentials)

    # In development mode, return demo user
    if settings.app.environment == "development":
        return AuthUser(
            id=UUID(DEMO_USER_ID),
            email="demo@voiceai.local",
            role="authenticated",
            aud="authenticated",
            exp=datetime.now(timezone.utc),
            iat=datetime.now(timezone.utc),
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )
