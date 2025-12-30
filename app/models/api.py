"""
API Key Models
"""
from datetime import datetime
from typing import Optional, List

from sqlalchemy import String, DateTime, ForeignKey, Boolean, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class APIKey(Base):
    """API Key for programmatic access."""

    __tablename__ = "api_keys"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    key_hash: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    key_prefix: Mapped[str] = mapped_column(String, nullable=False)  # first 8 chars
    permissions: Mapped[List] = mapped_column(JSONB, default=["*"])
    rate_limit_per_minute: Mapped[int] = mapped_column(Integer, default=1000)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="api_keys")
    usage_logs: Mapped[List["APIKeyUsage"]] = relationship(
        "APIKeyUsage", back_populates="api_key", cascade="all, delete-orphan"
    )


class APIKeyUsage(Base):
    """API Key usage log."""

    __tablename__ = "api_key_usage"

    api_key_id: Mapped[str] = mapped_column(
        ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=False, index=True
    )
    endpoint: Mapped[str] = mapped_column(String, nullable=False)
    method: Mapped[str] = mapped_column(String, nullable=False)
    status_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    response_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    api_key: Mapped["APIKey"] = relationship("APIKey", back_populates="usage_logs")


# Import for type hints
from app.models.workspace import Workspace  # noqa: E402
