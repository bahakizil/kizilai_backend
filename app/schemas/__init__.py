"""
Pydantic Schemas for API request/response validation.
"""
from app.schemas.analytics import (
    # Usage Daily
    AgentUsageDailyBase,
    AgentUsageDailyResponse,
    # Model Stats
    ModelUsageStatsBase,
    ModelUsageStatsResponse,
    # Overview
    AnalyticsOverview,
    UsageBreakdown,
    CostBreakdown,
    LatencyStats,
    QualityMetrics,
    # Time Series
    TimeSeriesPoint,
    TimeSeriesData,
    AnalyticsTimeSeries,
    # Agent Analytics
    AgentAnalyticsSummary,
    AgentAnalyticsDetail,
    # Prompt Versions
    PromptVersionBase,
    PromptVersionCreate,
    PromptVersionResponse,
    # Recordings
    CallRecordingBase,
    CallRecordingResponse,
    CallRecordingWithUrl,
    # Errors
    ErrorLogBase,
    ErrorLogResponse,
    # Message Metrics
    MessageMetricsResponse,
    # Export
    AnalyticsExportRequest,
    AnalyticsExportResponse,
)

__all__ = [
    # Usage Daily
    "AgentUsageDailyBase",
    "AgentUsageDailyResponse",
    # Model Stats
    "ModelUsageStatsBase",
    "ModelUsageStatsResponse",
    # Overview
    "AnalyticsOverview",
    "UsageBreakdown",
    "CostBreakdown",
    "LatencyStats",
    "QualityMetrics",
    # Time Series
    "TimeSeriesPoint",
    "TimeSeriesData",
    "AnalyticsTimeSeries",
    # Agent Analytics
    "AgentAnalyticsSummary",
    "AgentAnalyticsDetail",
    # Prompt Versions
    "PromptVersionBase",
    "PromptVersionCreate",
    "PromptVersionResponse",
    # Recordings
    "CallRecordingBase",
    "CallRecordingResponse",
    "CallRecordingWithUrl",
    # Errors
    "ErrorLogBase",
    "ErrorLogResponse",
    # Message Metrics
    "MessageMetricsResponse",
    # Export
    "AnalyticsExportRequest",
    "AnalyticsExportResponse",
]
