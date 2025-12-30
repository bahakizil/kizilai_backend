"""
Channel Models (Phone, Widget)
"""
from datetime import datetime
from typing import Optional, List

from sqlalchemy import String, DateTime, ForeignKey, Text, Boolean, Integer
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class PhoneNumber(Base):
    """Phone number for voice calls."""

    __tablename__ = "phone_numbers"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    agent_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("agents.id", ondelete="SET NULL"), nullable=True, index=True
    )
    phone_number: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    twilio_sid: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    country: Mapped[str] = mapped_column(String, default="TR")
    capabilities: Mapped[dict] = mapped_column(
        JSONB, default={"voice": True, "sms": False}
    )
    monthly_cost_cents: Mapped[int] = mapped_column(Integer, default=1500)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default="now()",
        onupdate="now()",
    )

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="phone_numbers")
    agent: Mapped[Optional["Agent"]] = relationship("Agent", back_populates="phone_numbers")
    calls: Mapped[List["Call"]] = relationship("Call", back_populates="phone_number")


class WebWidget(Base):
    """Embeddable web widget for voice calls."""

    __tablename__ = "web_widgets"

    agent_id: Mapped[str] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    button_color: Mapped[str] = mapped_column(String, default="#7C3AED")
    button_icon: Mapped[str] = mapped_column(String, default="microphone")
    button_position: Mapped[str] = mapped_column(String, default="bottom-right")
    button_size: Mapped[str] = mapped_column(String, default="medium")
    greeting_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    allowed_domains: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), nullable=True)
    custom_css: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default="now()",
        onupdate="now()",
    )

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", back_populates="web_widgets")
    calls: Mapped[List["Call"]] = relationship("Call", back_populates="widget")


# Import for type hints
from app.models.workspace import Workspace  # noqa: E402
from app.models.agent import Agent  # noqa: E402
from app.models.call import Call  # noqa: E402
