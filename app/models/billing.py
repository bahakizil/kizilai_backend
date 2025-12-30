"""
Billing & Usage Models
"""
from datetime import datetime, date
from typing import Optional

from sqlalchemy import String, DateTime, Date, ForeignKey, Text, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class UsageRecord(Base):
    """Daily usage record for billing."""

    __tablename__ = "usage_records"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)

    # Voice Usage
    stt_seconds: Mapped[float] = mapped_column(Float, default=0)
    tts_characters: Mapped[int] = mapped_column(Integer, default=0)
    phone_minutes: Mapped[float] = mapped_column(Float, default=0)

    # LLM Usage
    llm_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_output_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # Counts
    call_count: Mapped[int] = mapped_column(Integer, default=0)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    api_request_count: Mapped[int] = mapped_column(Integer, default=0)

    # Costs (in cents)
    stt_cost_cents: Mapped[int] = mapped_column(Integer, default=0)
    tts_cost_cents: Mapped[int] = mapped_column(Integer, default=0)
    llm_cost_cents: Mapped[int] = mapped_column(Integer, default=0)
    phone_cost_cents: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_cents: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="usage_records")


class Invoice(Base):
    """Invoice for billing."""

    __tablename__ = "invoices"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    stripe_invoice_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String, default="usd")
    status: Mapped[str] = mapped_column(String, default="draft")  # draft, open, paid, void
    period_start: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    period_end: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    pdf_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace")


# Import for type hints
from app.models.workspace import Workspace  # noqa: E402
