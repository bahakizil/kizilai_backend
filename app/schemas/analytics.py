"""
Analytics Schemas
Pydantic schemas for analytics data.
"""
from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, Field


# =============================================================================
# Usage Daily Schemas
# =============================================================================

class AgentUsageDailyBase(BaseModel):
    """Base schema for daily agent usage."""
    date: date
    total_calls: int = 0
    completed_calls: int = 0
    failed_calls: int = 0
    total_duration_seconds: float = 0
    avg_duration_seconds: float = 0

    # STT
    stt_provider: Optional[str] = None
    stt_model: Optional[str] = None
    stt_seconds: float = 0
    stt_cost_cents: int = 0

    # LLM
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_total_tokens: int = 0
    llm_cost_cents: int = 0

    # TTS
    tts_provider: Optional[str] = None
    tts_model: Optional[str] = None
    tts_characters: int = 0
    tts_cost_cents: int = 0

    # Latency
    avg_ttfa_ms: Optional[float] = None
    avg_stt_latency_ms: Optional[float] = None
    avg_llm_latency_ms: Optional[float] = None
    avg_tts_latency_ms: Optional[float] = None
    p95_ttfa_ms: Optional[float] = None
    p99_ttfa_ms: Optional[float] = None

    # Quality
    avg_sentiment_score: Optional[float] = None
    positive_calls: int = 0
    negative_calls: int = 0

    # Cost
    total_cost_cents: int = 0


class AgentUsageDailyResponse(AgentUsageDailyBase):
    """Response schema for daily agent usage."""
    id: str
    agent_id: str
    created_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Model Usage Stats Schemas
# =============================================================================

class ModelUsageStatsBase(BaseModel):
    """Base schema for model usage stats."""
    date: date
    service_type: str  # stt, llm, tts
    provider: str
    model: str

    request_count: int = 0
    total_duration_ms: float = 0
    avg_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None

    # For STT
    total_audio_seconds: float = 0

    # For LLM
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # For TTS
    total_characters: int = 0
    total_audio_generated_seconds: float = 0

    # Cost & Errors
    total_cost_cents: int = 0
    error_count: int = 0
    timeout_count: int = 0


class ModelUsageStatsResponse(ModelUsageStatsBase):
    """Response schema for model usage stats."""
    id: str
    workspace_id: str
    created_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Analytics Overview Schemas
# =============================================================================

class AnalyticsOverview(BaseModel):
    """Overview analytics for dashboard."""
    # Totals
    total_calls: int = 0
    total_duration_minutes: float = 0
    total_cost_cents: int = 0
    total_messages: int = 0

    # Averages
    avg_call_duration_seconds: float = 0
    avg_cost_per_call_cents: float = 0
    avg_messages_per_call: float = 0

    # Rates
    success_rate: float = 0  # 0-1
    completion_rate: float = 0  # 0-1

    # Latency
    avg_ttfa_ms: Optional[float] = None
    p95_ttfa_ms: Optional[float] = None

    # Period
    start_date: date
    end_date: date


class UsageBreakdown(BaseModel):
    """Usage breakdown by service type."""
    stt_seconds: float = 0
    stt_cost_cents: int = 0

    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_total_tokens: int = 0
    llm_cost_cents: int = 0

    tts_characters: int = 0
    tts_cost_cents: int = 0

    total_cost_cents: int = 0


class CostBreakdown(BaseModel):
    """Cost breakdown for billing."""
    stt_cost_cents: int = 0
    llm_cost_cents: int = 0
    tts_cost_cents: int = 0
    phone_cost_cents: int = 0
    total_cost_cents: int = 0

    # Percentages
    stt_percentage: float = 0
    llm_percentage: float = 0
    tts_percentage: float = 0
    phone_percentage: float = 0


class LatencyStats(BaseModel):
    """Latency statistics."""
    avg_ttfa_ms: Optional[float] = None
    p50_ttfa_ms: Optional[float] = None
    p95_ttfa_ms: Optional[float] = None
    p99_ttfa_ms: Optional[float] = None

    avg_stt_ms: Optional[float] = None
    avg_llm_ms: Optional[float] = None
    avg_tts_ms: Optional[float] = None


class QualityMetrics(BaseModel):
    """Quality metrics."""
    avg_sentiment_score: Optional[float] = None
    positive_calls_count: int = 0
    neutral_calls_count: int = 0
    negative_calls_count: int = 0

    avg_user_rating: Optional[float] = None
    rating_count: int = 0


# =============================================================================
# Time Series Schemas
# =============================================================================

class TimeSeriesPoint(BaseModel):
    """Single data point in a time series."""
    date: date
    value: float


class TimeSeriesData(BaseModel):
    """Time series data for charts."""
    label: str
    data: List[TimeSeriesPoint]
    total: float = 0


class AnalyticsTimeSeries(BaseModel):
    """Multiple time series for analytics charts."""
    calls: List[TimeSeriesPoint] = []
    duration: List[TimeSeriesPoint] = []
    cost: List[TimeSeriesPoint] = []
    messages: List[TimeSeriesPoint] = []


# =============================================================================
# Agent Analytics Schemas
# =============================================================================

class AgentAnalyticsSummary(BaseModel):
    """Summary analytics for a single agent."""
    agent_id: str
    agent_name: str

    total_calls: int = 0
    total_duration_minutes: float = 0
    total_cost_cents: int = 0
    avg_call_duration_seconds: float = 0
    success_rate: float = 0

    # Comparison (vs previous period)
    calls_change_percent: Optional[float] = None
    duration_change_percent: Optional[float] = None
    cost_change_percent: Optional[float] = None


class AgentAnalyticsDetail(AgentAnalyticsSummary):
    """Detailed analytics for a single agent."""
    usage: UsageBreakdown
    latency: LatencyStats
    quality: QualityMetrics
    daily_data: List[AgentUsageDailyResponse] = []


# =============================================================================
# Prompt Version Schemas
# =============================================================================

class PromptVersionBase(BaseModel):
    """Base schema for prompt versions."""
    system_prompt: str
    first_message: Optional[str] = None
    change_reason: Optional[str] = None


class PromptVersionCreate(PromptVersionBase):
    """Create schema for prompt versions."""
    pass


class PromptVersionResponse(PromptVersionBase):
    """Response schema for prompt versions."""
    id: str
    agent_id: str
    version: int
    is_active: bool

    total_calls: int = 0
    avg_sentiment_score: Optional[float] = None
    avg_duration_seconds: Optional[float] = None

    created_by: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Recording Schemas
# =============================================================================

class CallRecordingBase(BaseModel):
    """Base schema for call recordings."""
    recording_type: str = "full"  # full, user_only, agent_only


class CallRecordingResponse(BaseModel):
    """Response schema for call recordings."""
    id: str
    call_id: str
    recording_type: str
    storage_provider: str

    format: str
    sample_rate: int
    channels: int
    duration_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None

    status: str
    transcription_status: Optional[str] = None

    retention_days: int
    expires_at: Optional[datetime] = None

    created_at: datetime

    class Config:
        from_attributes = True


class CallRecordingWithUrl(CallRecordingResponse):
    """Recording response with download URL."""
    url: Optional[str] = None


# =============================================================================
# Error Log Schemas
# =============================================================================

class ErrorLogBase(BaseModel):
    """Base schema for error logs."""
    service_type: str
    error_message: str
    provider: Optional[str] = None
    model: Optional[str] = None
    error_code: Optional[str] = None
    error_details: Optional[dict] = None


class ErrorLogResponse(ErrorLogBase):
    """Response schema for error logs."""
    id: str
    workspace_id: Optional[str] = None
    agent_id: Optional[str] = None
    call_id: Optional[str] = None
    message_id: Optional[str] = None

    retry_count: int = 0
    was_recovered: bool = False
    stack_trace: Optional[str] = None

    created_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Message Metrics Schemas
# =============================================================================

class MessageMetricsResponse(BaseModel):
    """Response schema for message metrics."""
    id: str
    message_id: str
    call_id: str

    # STT
    stt_provider: Optional[str] = None
    stt_model: Optional[str] = None
    stt_duration_ms: Optional[float] = None
    stt_confidence: Optional[float] = None
    stt_words_count: Optional[int] = None

    # LLM
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_duration_ms: Optional[float] = None
    llm_input_tokens: Optional[int] = None
    llm_output_tokens: Optional[int] = None
    llm_total_tokens: Optional[int] = None

    # TTS
    tts_provider: Optional[str] = None
    tts_model: Optional[str] = None
    tts_duration_ms: Optional[float] = None
    tts_characters: Optional[int] = None
    tts_audio_duration_ms: Optional[float] = None

    # RAG
    rag_enabled: bool = False
    rag_chunks_retrieved: Optional[int] = None
    rag_query_ms: Optional[float] = None

    # Costs
    stt_cost_cents: float = 0
    llm_cost_cents: float = 0
    tts_cost_cents: float = 0
    total_cost_cents: float = 0

    created_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Export Schemas
# =============================================================================

class AnalyticsExportRequest(BaseModel):
    """Request schema for analytics export."""
    format: str = "csv"  # csv, json
    export_type: str = "calls"  # calls, messages, usage
    start_date: date
    end_date: date
    agent_ids: Optional[List[str]] = None


class AnalyticsExportResponse(BaseModel):
    """Response schema for analytics export."""
    download_url: str
    filename: str
    format: str
    record_count: int
    generated_at: datetime
