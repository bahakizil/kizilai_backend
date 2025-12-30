"""
API v1 Router
"""
from fastapi import APIRouter

from app.api.v1 import agents, workspaces, calls, auth, phone_numbers, api_keys, analytics

router = APIRouter()

# Include sub-routers
router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
router.include_router(workspaces.router, prefix="/workspaces", tags=["Workspaces"])
router.include_router(agents.router, prefix="/agents", tags=["Agents"])
router.include_router(calls.router, prefix="/calls", tags=["Calls"])
router.include_router(phone_numbers.router, prefix="/phone-numbers", tags=["Phone Numbers"])
router.include_router(api_keys.router, prefix="/api-keys", tags=["API Keys"])
router.include_router(analytics.router)
