"""
Authentication Endpoints - Supabase Auth Integration
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from gotrue.errors import AuthApiError

from app.core.supabase import supabase_admin, supabase
from app.core.auth import get_current_user, get_current_user_or_demo, AuthUser

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class RefreshRequest(BaseModel):
    refresh_token: str


class ResetPasswordRequest(BaseModel):
    email: EmailStr


class UpdatePasswordRequest(BaseModel):
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: Optional[str] = None


# =============================================================================
# Auth Endpoints
# =============================================================================

@router.post("/signup", response_model=TokenResponse)
async def signup(request: RegisterRequest):
    """
    Register a new user with email and password.
    Supabase will send a confirmation email if email verification is enabled.
    """
    try:
        response = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password,
            "options": {
                "data": {
                    "full_name": request.full_name,
                }
            }
        })

        if response.user is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create user",
            )

        session = response.session
        if session is None:
            # Email confirmation required
            return TokenResponse(
                access_token="",
                refresh_token="",
                expires_in=0,
                user={
                    "id": str(response.user.id),
                    "email": response.user.email,
                    "full_name": request.full_name,
                    "email_confirmed": False,
                },
            )

        return TokenResponse(
            access_token=session.access_token,
            refresh_token=session.refresh_token,
            expires_in=session.expires_in,
            user={
                "id": str(response.user.id),
                "email": response.user.email,
                "full_name": request.full_name,
                "email_confirmed": response.user.email_confirmed_at is not None,
            },
        )

    except AuthApiError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """
    Login with email and password.
    Returns access token and refresh token.
    """
    try:
        response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password,
        })

        if response.session is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        user_meta = response.user.user_metadata or {}

        return TokenResponse(
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            expires_in=response.session.expires_in,
            user={
                "id": str(response.user.id),
                "email": response.user.email,
                "full_name": user_meta.get("full_name"),
                "avatar_url": user_meta.get("avatar_url"),
            },
        )

    except AuthApiError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )


@router.post("/logout")
async def logout(current_user: AuthUser = Depends(get_current_user)):
    """
    Logout current user and invalidate session.
    """
    try:
        supabase.auth.sign_out()
        return {"message": "Successfully logged out"}
    except Exception as e:
        # Still return success even if sign out fails
        return {"message": "Logged out"}


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest):
    """
    Refresh access token using refresh token.
    """
    try:
        response = supabase.auth.refresh_session(request.refresh_token)

        if response.session is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        user_meta = response.user.user_metadata or {}

        return TokenResponse(
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            expires_in=response.session.expires_in,
            user={
                "id": str(response.user.id),
                "email": response.user.email,
                "full_name": user_meta.get("full_name"),
                "avatar_url": user_meta.get("avatar_url"),
            },
        )

    except AuthApiError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )


@router.post("/forgot-password")
async def forgot_password(request: ResetPasswordRequest):
    """
    Send password reset email.
    """
    try:
        supabase.auth.reset_password_email(request.email)
        return {"message": "Password reset email sent"}
    except AuthApiError as e:
        # Don't reveal if email exists
        return {"message": "If the email exists, a reset link has been sent"}


@router.post("/reset-password")
async def reset_password(
    request: UpdatePasswordRequest,
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Update password for authenticated user.
    Called after user clicks reset link in email.
    """
    try:
        response = supabase.auth.update_user({
            "password": request.password,
        })

        if response.user is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update password",
            )

        return {"message": "Password updated successfully"}

    except AuthApiError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: AuthUser = Depends(get_current_user_or_demo)):
    """
    Get current authenticated user info.
    """
    # Get user details from Supabase
    try:
        response = supabase_admin.auth.admin.get_user_by_id(str(current_user.id))

        if response.user is None:
            # Return basic info from token
            return UserResponse(
                id=str(current_user.id),
                email=current_user.email or "",
            )

        user_meta = response.user.user_metadata or {}

        return UserResponse(
            id=str(response.user.id),
            email=response.user.email,
            full_name=user_meta.get("full_name"),
            avatar_url=user_meta.get("avatar_url"),
            created_at=response.user.created_at,
        )

    except Exception:
        # Return basic info from token
        return UserResponse(
            id=str(current_user.id),
            email=current_user.email or "",
        )


@router.patch("/me", response_model=UserResponse)
async def update_me(
    full_name: Optional[str] = None,
    avatar_url: Optional[str] = None,
    current_user: AuthUser = Depends(get_current_user),
):
    """
    Update current user's profile.
    """
    try:
        update_data = {}
        if full_name is not None:
            update_data["full_name"] = full_name
        if avatar_url is not None:
            update_data["avatar_url"] = avatar_url

        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No update data provided",
            )

        response = supabase_admin.auth.admin.update_user_by_id(
            str(current_user.id),
            {"user_metadata": update_data},
        )

        if response.user is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update user",
            )

        user_meta = response.user.user_metadata or {}

        return UserResponse(
            id=str(response.user.id),
            email=response.user.email,
            full_name=user_meta.get("full_name"),
            avatar_url=user_meta.get("avatar_url"),
            created_at=response.user.created_at,
        )

    except AuthApiError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# =============================================================================
# OAuth (Social Login)
# =============================================================================

@router.get("/oauth/{provider}")
async def oauth_login(provider: str):
    """
    Get OAuth login URL for provider (google, github, etc.)
    Frontend should redirect user to this URL.
    """
    valid_providers = ["google", "github", "discord", "slack"]
    if provider not in valid_providers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid provider. Valid options: {', '.join(valid_providers)}",
        )

    try:
        response = supabase.auth.sign_in_with_oauth({
            "provider": provider,
            "options": {
                "redirect_to": f"http://localhost:3000/auth/callback",
            },
        })

        return {"url": response.url}

    except AuthApiError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
