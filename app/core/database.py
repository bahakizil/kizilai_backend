"""
Database Configuration - Supabase PostgreSQL with SQLAlchemy Async
"""
import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.models import Base

# Determine SSL settings based on environment
# Render and other cloud providers require SSL
is_production = os.getenv("APP_ENV", "development") == "production"
connect_args = {"ssl": "require"} if is_production else {"ssl": False}

# Create async engine using Supabase database
engine = create_async_engine(
    settings.supabase.database_url,
    echo=settings.app.debug,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    connect_args=connect_args,
)

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()
