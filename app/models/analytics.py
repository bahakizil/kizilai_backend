"""
Analytics & Metrics Models
Comprehensive tracking for usage, costs, recordings, and errors.
"""
from datetime import datetime, date
from typing import Optional, List

from sqlalchemy import String, DateTime, Date, ForeignKey, Text, Boolean, Float, Integer, BigInteger, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class AgentUsageDaily(Base):
    """Daily usage metrics aggregated per agent."""

    __tablename__ = "agent_usage_daily"
    __table_args__ = (
        UniqueConstraint("agent_id", "date", name="uq_agent_usage_daily_agent_date"),
        Index("idx_agent_usage_daily_agent", "agent_id"),
        Index("idx_agent_usage_daily_date", "date"),
    )

    agent_id: Mapped[str] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), nullable=False
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)

    # Call metrics
    total_calls: Mapped[int] = mapped_column(Integer, default=0)
    completed_calls: Mapped[int] = mapped_column(Integer, default=0)
    failed_calls: Mapped[int] = mapped_column(Integer, default=0)
    total_duration_seconds: Mapped[float] = mapped_column(Float, default=0)
    avg_duration_seconds: Mapped[float] = mapped_column(Float, default=0)

    # STT metrics
    stt_provider: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stt_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stt_seconds: Mapped[float] = mapped_column(Float, default=0)
    stt_cost_cents: Mapped[int] = mapped_column(Integer, default=0)

    # LLM metrics
    llm_provider: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    llm_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    llm_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_cost_cents: Mapped[int] = mapped_column(Integer, default=0)

    # TTS metrics
    tts_provider: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tts_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tts_characters: Mapped[int] = mapped_column(Integer, default=0)
    tts_cost_cents: Mapped[int] = mapped_column(Integer, default=0)

    # Latency metrics (averages)
    avg_ttfa_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_stt_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_llm_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_tts_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p95_ttfa_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p99_ttfa_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Quality metrics
    avg_sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    positive_calls: Mapped[int] = mapped_column(Integer, default=0)
    negative_calls: Mapped[int] = mapped_column(Integer, default=0)

    # Total cost
    total_cost_cents: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", back_populates="usage_daily")


class PromptVersion(Base):
    """Version history for agent system prompts."""

    __tablename__ = "prompt_versions"
    __table_args__ = (
        UniqueConstraint("agent_id", "version", name="uq_prompt_versions_agent_version"),
        Index("idx_prompt_versions_agent", "agent_id"),
    )

    agent_id: Mapped[str] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    # Prompt content
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    first_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    created_by: Mapped[Optional[str]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    change_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Performance tracking
    total_calls: Mapped[int] = mapped_column(Integer, default=0)
    avg_sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", back_populates="prompt_versions")


class MessageMetrics(Base):
    """Detailed metrics for each message in a call."""

    __tablename__ = "message_metrics"
    __table_args__ = (
        Index("idx_message_metrics_message", "message_id"),
        Index("idx_message_metrics_call", "call_id"),
    )

    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    call_id: Mapped[str] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False
    )

    # STT details
    stt_provider: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stt_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stt_start_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    stt_end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    stt_duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stt_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stt_is_final: Mapped[bool] = mapped_column(Boolean, default=True)
    stt_words_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # LLM details
    llm_provider: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    llm_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    llm_start_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    llm_end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    llm_duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    llm_input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    llm_output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    llm_total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    llm_temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    llm_prompt_cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)

    # TTS details
    tts_provider: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tts_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tts_voice_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tts_start_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    tts_end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    tts_duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tts_characters: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tts_audio_duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # RAG details
    rag_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    rag_chunks_retrieved: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rag_query_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rag_relevance_scores: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Function call details
    function_call_count: Mapped[int] = mapped_column(Integer, default=0)
    function_total_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Cost breakdown
    stt_cost_cents: Mapped[float] = mapped_column(Float, default=0)
    llm_cost_cents: Mapped[float] = mapped_column(Float, default=0)
    tts_cost_cents: Mapped[float] = mapped_column(Float, default=0)
    total_cost_cents: Mapped[float] = mapped_column(Float, default=0)

    # Relationships
    message: Mapped["Message"] = relationship("Message", back_populates="metrics")
    call: Mapped["Call"] = relationship("Call", back_populates="message_metrics")


class CallRecording(Base):
    """Audio recordings for calls."""

    __tablename__ = "call_recordings"
    __table_args__ = (
        Index("idx_call_recordings_call", "call_id"),
        Index("idx_call_recordings_expires", "expires_at"),
    )

    call_id: Mapped[str] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False
    )

    # Recording info
    recording_type: Mapped[str] = mapped_column(String, nullable=False)  # full, user_only, agent_only
    storage_provider: Mapped[str] = mapped_column(String, nullable=False)  # s3, supabase, gcs
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    storage_bucket: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Audio properties
    format: Mapped[str] = mapped_column(String, default="wav")  # wav, mp3, opus
    sample_rate: Mapped[int] = mapped_column(Integer, default=16000)
    channels: Mapped[int] = mapped_column(Integer, default=1)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    # Processing status
    status: Mapped[str] = mapped_column(String, default="pending")  # pending, uploaded, processed, deleted
    transcription_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # null, processing, completed, failed

    # Retention
    retention_days: Mapped[int] = mapped_column(Integer, default=90)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    call: Mapped["Call"] = relationship("Call", back_populates="recordings")


class ModelUsageStats(Base):
    """Aggregate usage statistics per model per day."""

    __tablename__ = "model_usage_stats"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "date", "service_type", "provider", "model",
            name="uq_model_usage_stats_workspace_date_service"
        ),
        Index("idx_model_usage_stats_workspace", "workspace_id"),
        Index("idx_model_usage_stats_date", "date"),
    )

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)

    # Model info
    service_type: Mapped[str] = mapped_column(String, nullable=False)  # stt, llm, tts
    provider: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)

    # Usage metrics
    request_count: Mapped[int] = mapped_column(Integer, default=0)
    total_duration_ms: Mapped[float] = mapped_column(Float, default=0)
    avg_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p95_latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # For STT
    total_audio_seconds: Mapped[float] = mapped_column(Float, default=0)

    # For LLM
    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # For TTS
    total_characters: Mapped[int] = mapped_column(Integer, default=0)
    total_audio_generated_seconds: Mapped[float] = mapped_column(Float, default=0)

    # Cost
    total_cost_cents: Mapped[int] = mapped_column(Integer, default=0)

    # Errors
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    timeout_count: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="model_usage_stats")


class ErrorLog(Base):
    """Error tracking for debugging and reliability."""

    __tablename__ = "error_logs"
    __table_args__ = (
        Index("idx_error_logs_workspace", "workspace_id"),
        Index("idx_error_logs_call", "call_id"),
        Index("idx_error_logs_created", "created_at"),
    )

    # Context
    workspace_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("workspaces.id", ondelete="SET NULL"), nullable=True
    )
    agent_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )
    call_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("calls.id", ondelete="SET NULL"), nullable=True
    )
    message_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )

    # Error info
    service_type: Mapped[str] = mapped_column(String, nullable=False)  # stt, llm, tts, rag, function
    provider: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Retry info
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    was_recovered: Mapped[bool] = mapped_column(Boolean, default=False)

    # Stack trace
    stack_trace: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


# Import for type hints
from app.models.agent import Agent  # noqa: E402
from app.models.workspace import Workspace  # noqa: E402
from app.models.call import Call, Message  # noqa: E402
