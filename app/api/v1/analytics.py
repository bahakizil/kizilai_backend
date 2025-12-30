"""
Analytics API Endpoints
Comprehensive analytics and metrics for workspace and agents.
"""
from datetime import date, datetime, timedelta
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.agent import Agent
from app.models.call import Call, Message
from app.models.analytics import (
    AgentUsageDaily,
    ModelUsageStats,
    MessageMetrics,
    CallRecording,
    ErrorLog,
    PromptVersion,
)
from app.models.billing import UsageRecord
from app.schemas.analytics import (
    AgentUsageDailyResponse,
    ModelUsageStatsResponse,
    AnalyticsOverview,
    UsageBreakdown,
    CostBreakdown,
    LatencyStats,
    QualityMetrics,
    TimeSeriesPoint,
    AnalyticsTimeSeries,
    AgentAnalyticsSummary,
    AgentAnalyticsDetail,
    PromptVersionResponse,
    PromptVersionCreate,
    CallRecordingResponse,
    CallRecordingWithUrl,
    ErrorLogResponse,
    MessageMetricsResponse,
)
from app.services.metrics_service import MetricsService
from app.services.recording_service import RecordingService

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Demo workspace ID (for development)
DEMO_WORKSPACE_ID = "00000000-0000-0000-0000-000000000001"


# =============================================================================
# Overview Endpoints
# =============================================================================

@router.get("/overview", response_model=AnalyticsOverview)
async def get_analytics_overview(
    start_date: date = Query(default=None, description="Start date (default: 30 days ago)"),
    end_date: date = Query(default=None, description="End date (default: today)"),
    agent_id: Optional[str] = Query(default=None, description="Filter by agent ID"),
    db: AsyncSession = Depends(get_db),
):
    """Get analytics overview for the workspace."""
    # Default date range
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    workspace_id = DEMO_WORKSPACE_ID

    # Build query
    filters = [
        Call.workspace_id == workspace_id,
        Call.started_at >= datetime.combine(start_date, datetime.min.time()),
        Call.started_at <= datetime.combine(end_date, datetime.max.time()),
    ]
    if agent_id:
        filters.append(Call.agent_id == agent_id)

    # Get aggregated data
    stmt = select(
        func.count(Call.id).label("total_calls"),
        func.sum(Call.duration_seconds).label("total_duration"),
        func.sum(Call.cost_cents).label("total_cost"),
        func.sum(Call.message_count).label("total_messages"),
        func.avg(Call.duration_seconds).label("avg_duration"),
        func.avg(Call.cost_cents).label("avg_cost"),
        func.avg(Call.message_count).label("avg_messages"),
        func.avg(Call.avg_ttfa_ms).label("avg_ttfa"),
    ).where(and_(*filters))

    result = await db.execute(stmt)
    row = result.one()

    # Get success rate
    completed_stmt = select(func.count(Call.id)).where(
        and_(*filters, Call.status == "completed")
    )
    completed_result = await db.execute(completed_stmt)
    completed_count = completed_result.scalar() or 0

    total_calls = row.total_calls or 0
    success_rate = completed_count / total_calls if total_calls > 0 else 0

    return AnalyticsOverview(
        total_calls=total_calls,
        total_duration_minutes=(row.total_duration or 0) / 60,
        total_cost_cents=int(row.total_cost or 0),
        total_messages=row.total_messages or 0,
        avg_call_duration_seconds=row.avg_duration or 0,
        avg_cost_per_call_cents=row.avg_cost or 0,
        avg_messages_per_call=row.avg_messages or 0,
        success_rate=success_rate,
        completion_rate=success_rate,
        avg_ttfa_ms=row.avg_ttfa,
        start_date=start_date,
        end_date=end_date,
    )


@router.get("/usage", response_model=UsageBreakdown)
async def get_usage_breakdown(
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Get usage breakdown by service type."""
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    workspace_id = DEMO_WORKSPACE_ID

    # Get usage from usage_records
    stmt = select(
        func.sum(UsageRecord.stt_seconds).label("stt_seconds"),
        func.sum(UsageRecord.llm_input_tokens).label("llm_input_tokens"),
        func.sum(UsageRecord.llm_output_tokens).label("llm_output_tokens"),
        func.sum(UsageRecord.tts_characters).label("tts_characters"),
        func.sum(UsageRecord.total_cost_cents).label("total_cost"),
    ).where(
        and_(
            UsageRecord.workspace_id == workspace_id,
            UsageRecord.date >= start_date,
            UsageRecord.date <= end_date,
        )
    )

    result = await db.execute(stmt)
    row = result.one()

    llm_input = row.llm_input_tokens or 0
    llm_output = row.llm_output_tokens or 0
    llm_total = llm_input + llm_output

    # Calculate approximate costs (simplified)
    stt_cost = int((row.stt_seconds or 0) * 0.0043)  # Deepgram rate
    llm_cost = int(llm_input / 1000 * 0.015 + llm_output / 1000 * 0.06)  # Approximate rate
    tts_cost = int((row.tts_characters or 0) / 1000 * 0.015)  # Cartesia rate

    return UsageBreakdown(
        stt_seconds=row.stt_seconds or 0,
        stt_cost_cents=stt_cost,
        llm_input_tokens=llm_input,
        llm_output_tokens=llm_output,
        llm_total_tokens=llm_total,
        llm_cost_cents=llm_cost,
        tts_characters=row.tts_characters or 0,
        tts_cost_cents=tts_cost,
        total_cost_cents=int(row.total_cost or 0),
    )


@router.get("/costs", response_model=CostBreakdown)
async def get_cost_breakdown(
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Get cost breakdown by service type."""
    usage = await get_usage_breakdown(start_date, end_date, db)

    total = usage.stt_cost_cents + usage.llm_cost_cents + usage.tts_cost_cents
    if total == 0:
        total = 1  # Avoid division by zero

    return CostBreakdown(
        stt_cost_cents=usage.stt_cost_cents,
        llm_cost_cents=usage.llm_cost_cents,
        tts_cost_cents=usage.tts_cost_cents,
        phone_cost_cents=0,
        total_cost_cents=total,
        stt_percentage=usage.stt_cost_cents / total * 100,
        llm_percentage=usage.llm_cost_cents / total * 100,
        tts_percentage=usage.tts_cost_cents / total * 100,
        phone_percentage=0,
    )


@router.get("/latency", response_model=LatencyStats)
async def get_latency_stats(
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
    agent_id: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Get latency statistics."""
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    workspace_id = DEMO_WORKSPACE_ID

    filters = [
        Call.workspace_id == workspace_id,
        Call.started_at >= datetime.combine(start_date, datetime.min.time()),
        Call.started_at <= datetime.combine(end_date, datetime.max.time()),
    ]
    if agent_id:
        filters.append(Call.agent_id == agent_id)

    stmt = select(
        func.avg(Call.avg_ttfa_ms).label("avg_ttfa"),
        func.avg(Call.avg_llm_ms).label("avg_llm"),
        func.avg(Call.avg_tts_ms).label("avg_tts"),
    ).where(and_(*filters))

    result = await db.execute(stmt)
    row = result.one()

    return LatencyStats(
        avg_ttfa_ms=row.avg_ttfa,
        p50_ttfa_ms=None,  # Would need percentile calculation
        p95_ttfa_ms=None,
        p99_ttfa_ms=None,
        avg_stt_ms=None,
        avg_llm_ms=row.avg_llm,
        avg_tts_ms=row.avg_tts,
    )


@router.get("/time-series", response_model=AnalyticsTimeSeries)
async def get_analytics_time_series(
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
    granularity: str = Query(default="daily", description="daily, weekly, monthly"),
    db: AsyncSession = Depends(get_db),
):
    """Get time series data for charts."""
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    workspace_id = DEMO_WORKSPACE_ID

    # Get daily data from usage_records
    stmt = select(UsageRecord).where(
        and_(
            UsageRecord.workspace_id == workspace_id,
            UsageRecord.date >= start_date,
            UsageRecord.date <= end_date,
        )
    ).order_by(UsageRecord.date)

    result = await db.execute(stmt)
    records = result.scalars().all()

    # Also get call counts per day
    call_stmt = select(
        func.date(Call.started_at).label("call_date"),
        func.count(Call.id).label("count"),
        func.sum(Call.duration_seconds).label("duration"),
        func.sum(Call.cost_cents).label("cost"),
        func.sum(Call.message_count).label("messages"),
    ).where(
        and_(
            Call.workspace_id == workspace_id,
            Call.started_at >= datetime.combine(start_date, datetime.min.time()),
            Call.started_at <= datetime.combine(end_date, datetime.max.time()),
        )
    ).group_by(func.date(Call.started_at)).order_by(func.date(Call.started_at))

    call_result = await db.execute(call_stmt)
    call_data = {row.call_date: row for row in call_result.all()}

    # Build time series
    calls = []
    duration = []
    cost = []
    messages = []

    current = start_date
    while current <= end_date:
        call_row = call_data.get(current)
        calls.append(TimeSeriesPoint(date=current, value=call_row.count if call_row else 0))
        duration.append(TimeSeriesPoint(date=current, value=(call_row.duration or 0) / 60 if call_row else 0))
        cost.append(TimeSeriesPoint(date=current, value=(call_row.cost or 0) / 100 if call_row else 0))
        messages.append(TimeSeriesPoint(date=current, value=call_row.messages if call_row else 0))
        current += timedelta(days=1)

    return AnalyticsTimeSeries(
        calls=calls,
        duration=duration,
        cost=cost,
        messages=messages,
    )


# =============================================================================
# Model Usage Endpoints
# =============================================================================

@router.get("/models", response_model=List[ModelUsageStatsResponse])
async def get_model_usage_stats(
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
    service_type: Optional[str] = Query(default=None, description="stt, llm, tts"),
    db: AsyncSession = Depends(get_db),
):
    """Get usage statistics by model."""
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    workspace_id = DEMO_WORKSPACE_ID

    filters = [
        ModelUsageStats.workspace_id == workspace_id,
        ModelUsageStats.date >= start_date,
        ModelUsageStats.date <= end_date,
    ]
    if service_type:
        filters.append(ModelUsageStats.service_type == service_type)

    stmt = select(ModelUsageStats).where(and_(*filters)).order_by(ModelUsageStats.date.desc())

    result = await db.execute(stmt)
    return result.scalars().all()


# =============================================================================
# Agent Analytics Endpoints
# =============================================================================

@router.get("/agents", response_model=List[AgentAnalyticsSummary])
async def get_agents_analytics(
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Get analytics summary for all agents."""
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    workspace_id = DEMO_WORKSPACE_ID

    # Get all agents for workspace
    agents_stmt = select(Agent).where(Agent.workspace_id == workspace_id)
    agents_result = await db.execute(agents_stmt)
    agents = agents_result.scalars().all()

    summaries = []
    for agent in agents:
        # Get stats for this agent
        stats_stmt = select(
            func.count(Call.id).label("total_calls"),
            func.sum(Call.duration_seconds).label("total_duration"),
            func.sum(Call.cost_cents).label("total_cost"),
            func.avg(Call.duration_seconds).label("avg_duration"),
        ).where(
            and_(
                Call.agent_id == agent.id,
                Call.started_at >= datetime.combine(start_date, datetime.min.time()),
                Call.started_at <= datetime.combine(end_date, datetime.max.time()),
            )
        )

        stats_result = await db.execute(stats_stmt)
        stats = stats_result.one()

        # Get success rate
        completed_stmt = select(func.count(Call.id)).where(
            and_(
                Call.agent_id == agent.id,
                Call.status == "completed",
                Call.started_at >= datetime.combine(start_date, datetime.min.time()),
                Call.started_at <= datetime.combine(end_date, datetime.max.time()),
            )
        )
        completed_result = await db.execute(completed_stmt)
        completed_count = completed_result.scalar() or 0

        total_calls = stats.total_calls or 0
        success_rate = completed_count / total_calls if total_calls > 0 else 0

        summaries.append(AgentAnalyticsSummary(
            agent_id=str(agent.id),
            agent_name=agent.name,
            total_calls=total_calls,
            total_duration_minutes=(stats.total_duration or 0) / 60,
            total_cost_cents=int(stats.total_cost or 0),
            avg_call_duration_seconds=stats.avg_duration or 0,
            success_rate=success_rate,
        ))

    return summaries


@router.get("/agents/{agent_id}/daily", response_model=List[AgentUsageDailyResponse])
async def get_agent_daily_usage(
    agent_id: str,
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Get daily usage for a specific agent."""
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # Verify agent exists
    agent = await db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    stmt = select(AgentUsageDaily).where(
        and_(
            AgentUsageDaily.agent_id == agent_id,
            AgentUsageDaily.date >= start_date,
            AgentUsageDaily.date <= end_date,
        )
    ).order_by(AgentUsageDaily.date)

    result = await db.execute(stmt)
    return result.scalars().all()


# =============================================================================
# Prompt Version Endpoints
# =============================================================================

@router.get("/agents/{agent_id}/prompts", response_model=List[PromptVersionResponse])
async def get_agent_prompt_versions(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get prompt version history for an agent."""
    # Verify agent exists
    agent = await db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    stmt = select(PromptVersion).where(
        PromptVersion.agent_id == agent_id
    ).order_by(PromptVersion.version.desc())

    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("/agents/{agent_id}/prompts", response_model=PromptVersionResponse)
async def create_prompt_version(
    agent_id: str,
    prompt: PromptVersionCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new prompt version."""
    # Verify agent exists
    agent = await db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    metrics_service = MetricsService(db)
    version = await metrics_service.create_prompt_version(
        agent_id=UUID(agent_id),
        system_prompt=prompt.system_prompt,
        first_message=prompt.first_message,
        created_by=None,
        change_reason=prompt.change_reason,
    )

    await db.commit()
    return version


# =============================================================================
# Recording Endpoints
# =============================================================================

@router.get("/calls/{call_id}/recordings", response_model=List[CallRecordingWithUrl])
async def get_call_recordings(
    call_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get recordings for a call."""
    # Verify call exists
    call = await db.get(Call, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    recording_service = RecordingService(db)
    recordings = await recording_service.get_call_recordings(UUID(call_id))

    # Add URLs to recordings
    results = []
    for recording in recordings:
        url = await recording_service.get_recording_url(UUID(recording.id))
        results.append(CallRecordingWithUrl(
            id=recording.id,
            call_id=recording.call_id,
            recording_type=recording.recording_type,
            storage_provider=recording.storage_provider,
            format=recording.format,
            sample_rate=recording.sample_rate,
            channels=recording.channels,
            duration_seconds=recording.duration_seconds,
            file_size_bytes=recording.file_size_bytes,
            status=recording.status,
            transcription_status=recording.transcription_status,
            retention_days=recording.retention_days,
            expires_at=recording.expires_at,
            created_at=recording.created_at,
            url=url,
        ))

    return results


# =============================================================================
# Error Log Endpoints
# =============================================================================

@router.get("/errors", response_model=List[ErrorLogResponse])
async def get_error_logs(
    start_date: date = Query(default=None),
    end_date: date = Query(default=None),
    service_type: Optional[str] = Query(default=None),
    limit: int = Query(default=100, le=1000),
    db: AsyncSession = Depends(get_db),
):
    """Get error logs."""
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=7)

    workspace_id = DEMO_WORKSPACE_ID

    filters = [
        ErrorLog.workspace_id == workspace_id,
        ErrorLog.created_at >= datetime.combine(start_date, datetime.min.time()),
        ErrorLog.created_at <= datetime.combine(end_date, datetime.max.time()),
    ]
    if service_type:
        filters.append(ErrorLog.service_type == service_type)

    stmt = select(ErrorLog).where(
        and_(*filters)
    ).order_by(ErrorLog.created_at.desc()).limit(limit)

    result = await db.execute(stmt)
    return result.scalars().all()


# =============================================================================
# Message Metrics Endpoints
# =============================================================================

@router.get("/calls/{call_id}/metrics", response_model=List[MessageMetricsResponse])
async def get_call_message_metrics(
    call_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get detailed metrics for all messages in a call."""
    # Verify call exists
    call = await db.get(Call, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    stmt = select(MessageMetrics).where(
        MessageMetrics.call_id == call_id
    ).order_by(MessageMetrics.created_at)

    result = await db.execute(stmt)
    return result.scalars().all()
