"""
Supabase Client Configuration
Provides both admin (service_role) and public (anon) clients
"""
from functools import lru_cache

from supabase import create_client, Client

from app.core.config import settings


@lru_cache()
def get_supabase_admin() -> Client:
    """
    Get Supabase admin client with service_role key.
    Use this for server-side operations that need to bypass RLS.
    """
    return create_client(
        supabase_url=settings.supabase.url,
        supabase_key=settings.supabase.service_role_key,
    )


@lru_cache()
def get_supabase_public() -> Client:
    """
    Get Supabase public client with anon key.
    Use this for operations that should respect RLS.
    """
    return create_client(
        supabase_url=settings.supabase.url,
        supabase_key=settings.supabase.anon_key,
    )


def get_supabase_client_for_user(access_token: str) -> Client:
    """
    Get Supabase client authenticated as a specific user.
    Use this when you need to make requests on behalf of a user.
    """
    client = create_client(
        supabase_url=settings.supabase.url,
        supabase_key=settings.supabase.anon_key,
    )
    # Set the access token for authenticated requests
    client.auth.set_session(access_token, "")
    return client


# Convenience instances
supabase_admin = get_supabase_admin()
supabase = get_supabase_public()
