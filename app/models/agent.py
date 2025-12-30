"""
Agent & Knowledge Base Models
"""
from datetime import datetime
from typing import Optional, List, Any

from sqlalchemy import String, DateTime, ForeignKey, Text, Boolean, Float, Integer
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.models.base import Base


class Agent(Base):
    """Voice AI Agent model."""

    __tablename__ = "agents"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Basic Info
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String, default="draft")  # draft, active, paused

    # Personality
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    first_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    language: Mapped[str] = mapped_column(String, default="tr")

    # Voice Configuration
    voice_provider: Mapped[str] = mapped_column(String, default="cartesia")
    voice_id: Mapped[str] = mapped_column(String, default="leyla")
    voice_speed: Mapped[float] = mapped_column(Float, default=1.0)
    voice_pitch: Mapped[float] = mapped_column(Float, default=1.0)
    voice_emotion: Mapped[str] = mapped_column(String, default="neutral")

    # STT Configuration
    stt_provider: Mapped[str] = mapped_column(String, default="deepgram")
    stt_model: Mapped[str] = mapped_column(String, default="nova-3")
    stt_language: Mapped[str] = mapped_column(String, default="tr")

    # LLM Configuration
    llm_provider: Mapped[str] = mapped_column(String, default="openai")
    llm_model: Mapped[str] = mapped_column(String, default="gpt-4o-mini")
    llm_temperature: Mapped[float] = mapped_column(Float, default=0.7)
    llm_max_tokens: Mapped[int] = mapped_column(Integer, default=1024)

    # TTS Configuration
    tts_provider: Mapped[str] = mapped_column(String, default="cartesia")
    tts_model: Mapped[str] = mapped_column(String, default="sonic-3")

    # Advanced Settings
    max_duration_seconds: Mapped[int] = mapped_column(Integer, default=600)
    silence_timeout_seconds: Mapped[int] = mapped_column(Integer, default=30)
    interrupt_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Denormalized Stats
    total_calls: Mapped[int] = mapped_column(Integer, default=0)
    total_minutes: Mapped[float] = mapped_column(Float, default=0)
    avg_duration_seconds: Mapped[float] = mapped_column(Float, default=0)
    success_rate: Mapped[float] = mapped_column(Float, default=0)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default="now()",
        onupdate="now()",
    )

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="agents")
    knowledge_bases: Mapped[List["KnowledgeBase"]] = relationship(
        "KnowledgeBase", back_populates="agent", cascade="all, delete-orphan"
    )
    functions: Mapped[List["AgentFunction"]] = relationship(
        "AgentFunction", back_populates="agent", cascade="all, delete-orphan"
    )
    phone_numbers: Mapped[List["PhoneNumber"]] = relationship(
        "PhoneNumber", back_populates="agent"
    )
    web_widgets: Mapped[List["WebWidget"]] = relationship(
        "WebWidget", back_populates="agent", cascade="all, delete-orphan"
    )
    calls: Mapped[List["Call"]] = relationship("Call", back_populates="agent")
    usage_daily: Mapped[List["AgentUsageDaily"]] = relationship(
        "AgentUsageDaily", back_populates="agent", cascade="all, delete-orphan"
    )
    prompt_versions: Mapped[List["PromptVersion"]] = relationship(
        "PromptVersion", back_populates="agent", cascade="all, delete-orphan"
    )


class KnowledgeBase(Base):
    """Knowledge Base for RAG."""

    __tablename__ = "knowledge_bases"

    agent_id: Mapped[str] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default="now()",
        onupdate="now()",
    )

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", back_populates="knowledge_bases")
    documents: Mapped[List["Document"]] = relationship(
        "Document", back_populates="knowledge_base", cascade="all, delete-orphan"
    )


class Document(Base):
    """Document in Knowledge Base."""

    __tablename__ = "documents"

    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # pdf, docx, url, manual
    source_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    file_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending, processing, ready, failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default="now()",
        onupdate="now()",
    )

    # Relationships
    knowledge_base: Mapped["KnowledgeBase"] = relationship("KnowledgeBase", back_populates="documents")
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    """Vector-indexed chunk for RAG."""

    __tablename__ = "document_chunks"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector(384), nullable=True)  # all-MiniLM-L6-v2
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chunk_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chunk_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")


class AgentFunction(Base):
    """Custom function/tool for an agent."""

    __tablename__ = "agent_functions"

    agent_id: Mapped[str] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    parameters: Mapped[dict] = mapped_column(JSONB, nullable=False)  # JSON Schema
    webhook_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    webhook_method: Mapped[str] = mapped_column(String, default="POST")
    webhook_headers: Mapped[dict] = mapped_column(JSONB, default=dict)
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=30)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default="now()",
        onupdate="now()",
    )

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", back_populates="functions")


# Import for type hints
from app.models.workspace import Workspace  # noqa: E402
from app.models.channel import PhoneNumber, WebWidget  # noqa: E402
from app.models.call import Call  # noqa: E402
from app.models.analytics import AgentUsageDaily, PromptVersion  # noqa: E402
