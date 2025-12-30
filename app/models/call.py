"""
Call & Message Models
"""
from datetime import datetime
from typing import Optional, List

from sqlalchemy import String, DateTime, ForeignKey, Text, Boolean, Float, Integer
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Call(Base):
    """Voice call session."""

    __tablename__ = "calls"

    agent_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("agents.id", ondelete="SET NULL"), nullable=True, index=True
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Source
    channel: Mapped[str] = mapped_column(String, nullable=False)  # web, phone, api
    phone_number_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("phone_numbers.id"), nullable=True
    )
    widget_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("web_widgets.id"), nullable=True
    )

    # Caller Info
    caller_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    caller_phone: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    caller_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default="now()"
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String, default="active")  # active, completed, failed, transferred
    end_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # user_hangup, agent_hangup, timeout, error

    # Metrics
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    stt_seconds: Mapped[float] = mapped_column(Float, default=0)
    llm_tokens: Mapped[int] = mapped_column(Integer, default=0)
    tts_characters: Mapped[int] = mapped_column(Integer, default=0)
    avg_ttfa_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_llm_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_tts_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Cost
    cost_cents: Mapped[int] = mapped_column(Integer, default=0)

    # Quality
    user_rating: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1-5
    sentiment: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # positive, neutral, negative
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Metadata
    call_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    tags: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), nullable=True)

    # Summary
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    key_points: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Relationships
    agent: Mapped[Optional["Agent"]] = relationship("Agent", back_populates="calls")
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="calls")
    phone_number: Mapped[Optional["PhoneNumber"]] = relationship("PhoneNumber", back_populates="calls")
    widget: Mapped[Optional["WebWidget"]] = relationship("WebWidget", back_populates="calls")
    messages: Mapped[List["Message"]] = relationship(
        "Message", back_populates="call", cascade="all, delete-orphan"
    )
    message_metrics: Mapped[List["MessageMetrics"]] = relationship(
        "MessageMetrics", back_populates="call", cascade="all, delete-orphan"
    )
    recordings: Mapped[List["CallRecording"]] = relationship(
        "CallRecording", back_populates="call", cascade="all, delete-orphan"
    )


class Message(Base):
    """Individual message in a call."""

    __tablename__ = "messages"

    call_id: Mapped[str] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )

    role: Mapped[str] = mapped_column(String, nullable=False)  # user, assistant, system, function
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    audio_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Latency Metrics
    stt_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    llm_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tts_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ttfa_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Time to First Audio

    # Token details
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Provider info
    stt_provider: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stt_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    llm_provider: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    llm_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tts_provider: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tts_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tts_voice_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Audio info
    user_audio_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    agent_audio_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    user_audio_duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    agent_audio_duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Cost breakdown
    stt_cost_cents: Mapped[float] = mapped_column(Float, default=0)
    llm_cost_cents: Mapped[float] = mapped_column(Float, default=0)
    tts_cost_cents: Mapped[float] = mapped_column(Float, default=0)
    total_cost_cents: Mapped[float] = mapped_column(Float, default=0)

    # Confidence & quality
    stt_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    user_sentiment: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # positive, neutral, negative

    # Function Call
    function_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    function_args: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    function_result: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # RAG Context
    rag_chunks: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    rag_query: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Interruption
    was_interrupted: Mapped[bool] = mapped_column(Boolean, default=False)
    interrupted_at_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    call: Mapped["Call"] = relationship("Call", back_populates="messages")
    metrics: Mapped[Optional["MessageMetrics"]] = relationship(
        "MessageMetrics", back_populates="message", uselist=False
    )


# Import for type hints
from app.models.agent import Agent  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.models.channel import PhoneNumber, WebWidget  # noqa: E402
from app.models.analytics import MessageMetrics, CallRecording  # noqa: E402
