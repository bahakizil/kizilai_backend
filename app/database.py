"""
Database Connection - PostgreSQL with asyncpg
"""
import asyncpg
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator
from .config import get_settings

settings = get_settings()

# Connection pool
_pool: Optional[asyncpg.Pool] = None


async def init_db():
    """Initialize database connection pool."""
    global _pool
    _pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=2,
        max_size=10,
    )
    print(f"Database connected: {settings.database_url}")


async def close_db():
    """Close database connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        print("Database connection closed")


@asynccontextmanager
async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    """Get database connection from pool."""
    async with _pool.acquire() as conn:
        yield conn


async def fetch_all(query: str, *args):
    """Execute query and fetch all results."""
    async with get_db() as conn:
        return await conn.fetch(query, *args)


async def fetch_one(query: str, *args):
    """Execute query and fetch one result."""
    async with get_db() as conn:
        return await conn.fetchrow(query, *args)


async def execute(query: str, *args):
    """Execute query without returning results."""
    async with get_db() as conn:
        return await conn.execute(query, *args)
