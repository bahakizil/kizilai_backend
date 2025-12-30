"""
Metrics Service
Centralized metrics collection, storage, and aggregation.
"""
from datetime import datetime, date, timedelta
from typing import Optional, List
from uuid import UUID
from dataclasses import dataclass

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from app.models.analytics import (
    AgentUsageDaily,
    MessageMetrics,
    ModelUsageStats,
    ErrorLog,
    PromptVersion,
)
from app.models.call import Call, Message
from app.models.agent import Agent
from app.services.cost_calculator import CostCalculator


@dataclass
class STTMetrics:
    """STT metrics for a message."""
    provider: str
    model: str
    duration_ms: float
    confidence: Optional[float] = None
    words_count: Optional[int] = None
    is_final: bool = True


@dataclass
class LLMMetrics:
    """LLM metrics for a message."""
    provider: str
    model: str
    duration_ms: float
    input_tokens: int
    output_tokens: int
    temperature: Optional[float] = None
    prompt_cache_hit: bool = False


@dataclass
class TTSMetrics:
    """TTS metrics for a message."""
    provider: str
    model: str
    voice_id: str
    duration_ms: float
    characters: int
    audio_duration_ms: Optional[float] = None


@dataclass
class RAGMetrics:
    """RAG metrics for a message."""
    enabled: bool = False
    chunks_retrieved: Optional[int] = None
    query_ms: Optional[float] = None
    relevance_scores: Optional[dict] = None


@dataclass
class CallMetrics:
    """Final call metrics."""
    duration_seconds: float
    message_count: int
    stt_seconds: float
    llm_total_tokens: int
    tts_characters: int
    avg_ttfa_ms: Optional[float] = None
    avg_stt_ms: Optional[float] = None
    avg_llm_ms: Optional[float] = None
    avg_tts_ms: Optional[float] = None
    total_cost_cents: float = 0
    status: str = "completed"
    end_reason: Optional[str] = None


class MetricsService:
    """Centralized metrics collection and storage."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.cost_calculator = CostCalculator

    async def record_message_metrics(
        self,
        message_id: UUID,
        call_id: UUID,
        stt_metrics: Optional[STTMetrics] = None,
        llm_metrics: Optional[LLMMetrics] = None,
        tts_metrics: Optional[TTSMetrics] = None,
        rag_metrics: Optional[RAGMetrics] = None,
    ) -> MessageMetrics:
        """
        Record detailed metrics for a single message.

        This is called after each message cycle in the voice pipeline.
        """
        # Calculate costs
        cost = self.cost_calculator.calculate_message_cost(
            stt_provider=stt_metrics.provider if stt_metrics else None,
            stt_model=stt_metrics.model if stt_metrics else None,
            audio_seconds=stt_metrics.duration_ms / 1000 if stt_metrics else 0,
            llm_provider=llm_metrics.provider if llm_metrics else None,
            llm_model=llm_metrics.model if llm_metrics else None,
            input_tokens=llm_metrics.input_tokens if llm_metrics else 0,
            output_tokens=llm_metrics.output_tokens if llm_metrics else 0,
            tts_provider=tts_metrics.provider if tts_metrics else None,
            tts_model=tts_metrics.model if tts_metrics else None,
            characters=tts_metrics.characters if tts_metrics else 0,
        )

        now = datetime.utcnow()

        metrics = MessageMetrics(
            message_id=str(message_id),
            call_id=str(call_id),
            # STT details
            stt_provider=stt_metrics.provider if stt_metrics else None,
            stt_model=stt_metrics.model if stt_metrics else None,
            stt_start_time=now - timedelta(milliseconds=stt_metrics.duration_ms) if stt_metrics else None,
            stt_end_time=now if stt_metrics else None,
            stt_duration_ms=stt_metrics.duration_ms if stt_metrics else None,
            stt_confidence=stt_metrics.confidence if stt_metrics else None,
            stt_is_final=stt_metrics.is_final if stt_metrics else True,
            stt_words_count=stt_metrics.words_count if stt_metrics else None,
            # LLM details
            llm_provider=llm_metrics.provider if llm_metrics else None,
            llm_model=llm_metrics.model if llm_metrics else None,
            llm_start_time=now - timedelta(milliseconds=llm_metrics.duration_ms) if llm_metrics else None,
            llm_end_time=now if llm_metrics else None,
            llm_duration_ms=llm_metrics.duration_ms if llm_metrics else None,
            llm_input_tokens=llm_metrics.input_tokens if llm_metrics else None,
            llm_output_tokens=llm_metrics.output_tokens if llm_metrics else None,
            llm_total_tokens=(llm_metrics.input_tokens + llm_metrics.output_tokens) if llm_metrics else None,
            llm_temperature=llm_metrics.temperature if llm_metrics else None,
            llm_prompt_cache_hit=llm_metrics.prompt_cache_hit if llm_metrics else False,
            # TTS details
            tts_provider=tts_metrics.provider if tts_metrics else None,
            tts_model=tts_metrics.model if tts_metrics else None,
            tts_voice_id=tts_metrics.voice_id if tts_metrics else None,
            tts_start_time=now - timedelta(milliseconds=tts_metrics.duration_ms) if tts_metrics else None,
            tts_end_time=now if tts_metrics else None,
            tts_duration_ms=tts_metrics.duration_ms if tts_metrics else None,
            tts_characters=tts_metrics.characters if tts_metrics else None,
            tts_audio_duration_ms=tts_metrics.audio_duration_ms if tts_metrics else None,
            # RAG details
            rag_enabled=rag_metrics.enabled if rag_metrics else False,
            rag_chunks_retrieved=rag_metrics.chunks_retrieved if rag_metrics else None,
            rag_query_ms=rag_metrics.query_ms if rag_metrics else None,
            rag_relevance_scores=rag_metrics.relevance_scores if rag_metrics else None,
            # Costs
            stt_cost_cents=cost.stt_cost_cents,
            llm_cost_cents=cost.llm_cost_cents,
            tts_cost_cents=cost.tts_cost_cents,
            total_cost_cents=cost.total_cost_cents,
        )

        self.db.add(metrics)
        await self.db.flush()

        # Also update the message with provider info and costs
        message = await self.db.get(Message, str(message_id))
        if message:
            if stt_metrics:
                message.stt_provider = stt_metrics.provider
                message.stt_model = stt_metrics.model
                message.stt_confidence = stt_metrics.confidence
            if llm_metrics:
                message.llm_provider = llm_metrics.provider
                message.llm_model = llm_metrics.model
                message.input_tokens = llm_metrics.input_tokens
                message.output_tokens = llm_metrics.output_tokens
            if tts_metrics:
                message.tts_provider = tts_metrics.provider
                message.tts_model = tts_metrics.model
                message.tts_voice_id = tts_metrics.voice_id

            message.stt_cost_cents = cost.stt_cost_cents
            message.llm_cost_cents = cost.llm_cost_cents
            message.tts_cost_cents = cost.tts_cost_cents
            message.total_cost_cents = cost.total_cost_cents

        return metrics

    async def record_call_completion(
        self,
        call_id: UUID,
        final_metrics: CallMetrics,
    ) -> None:
        """
        Record final call metrics and update aggregates.

        This is called when a call ends.
        """
        call = await self.db.get(Call, str(call_id))
        if not call:
            return

        # Update call with final metrics
        call.duration_seconds = int(final_metrics.duration_seconds)
        call.message_count = final_metrics.message_count
        call.stt_seconds = final_metrics.stt_seconds
        call.llm_tokens = final_metrics.llm_total_tokens
        call.tts_characters = final_metrics.tts_characters
        call.avg_ttfa_ms = final_metrics.avg_ttfa_ms
        call.avg_llm_ms = final_metrics.avg_llm_ms
        call.avg_tts_ms = final_metrics.avg_tts_ms
        call.cost_cents = int(final_metrics.total_cost_cents)
        call.status = final_metrics.status
        call.end_reason = final_metrics.end_reason
        call.ended_at = datetime.utcnow()

        # Update agent stats
        if call.agent_id:
            agent = await self.db.get(Agent, call.agent_id)
            if agent:
                agent.total_calls += 1
                agent.total_minutes += final_metrics.duration_seconds / 60

                # Recalculate average duration
                if agent.total_calls > 0:
                    # Simple moving average approximation
                    agent.avg_duration_seconds = (
                        (agent.avg_duration_seconds * (agent.total_calls - 1) +
                         final_metrics.duration_seconds) / agent.total_calls
                    )

        # Update daily agent usage
        await self.update_daily_agent_usage(
            agent_id=UUID(call.agent_id) if call.agent_id else None,
            workspace_id=UUID(call.workspace_id),
            call_metrics=final_metrics,
            stt_provider=call.messages[0].stt_provider if call.messages else None,
            stt_model=call.messages[0].stt_model if call.messages else None,
            llm_provider=call.messages[0].llm_provider if call.messages else None,
            llm_model=call.messages[0].llm_model if call.messages else None,
            tts_provider=call.messages[0].tts_provider if call.messages else None,
            tts_model=call.messages[0].tts_model if call.messages else None,
        )

        await self.db.flush()

    async def update_daily_agent_usage(
        self,
        agent_id: Optional[UUID],
        workspace_id: UUID,
        call_metrics: CallMetrics,
        stt_provider: Optional[str] = None,
        stt_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        tts_provider: Optional[str] = None,
        tts_model: Optional[str] = None,
    ) -> None:
        """Update or create daily usage record for agent."""
        if not agent_id:
            return

        today = date.today()

        # Try to find existing record
        stmt = select(AgentUsageDaily).where(
            and_(
                AgentUsageDaily.agent_id == str(agent_id),
                AgentUsageDaily.date == today,
            )
        )
        result = await self.db.execute(stmt)
        usage = result.scalar_one_or_none()

        if usage:
            # Update existing record
            usage.total_calls += 1
            if call_metrics.status == "completed":
                usage.completed_calls += 1
            else:
                usage.failed_calls += 1
            usage.total_duration_seconds += call_metrics.duration_seconds
            usage.stt_seconds += call_metrics.stt_seconds
            usage.llm_total_tokens += call_metrics.llm_total_tokens
            usage.tts_characters += call_metrics.tts_characters
            usage.total_cost_cents += int(call_metrics.total_cost_cents)

            # Recalculate averages
            if usage.total_calls > 0:
                usage.avg_duration_seconds = usage.total_duration_seconds / usage.total_calls
        else:
            # Create new record
            usage = AgentUsageDaily(
                agent_id=str(agent_id),
                date=today,
                total_calls=1,
                completed_calls=1 if call_metrics.status == "completed" else 0,
                failed_calls=0 if call_metrics.status == "completed" else 1,
                total_duration_seconds=call_metrics.duration_seconds,
                avg_duration_seconds=call_metrics.duration_seconds,
                stt_provider=stt_provider,
                stt_model=stt_model,
                stt_seconds=call_metrics.stt_seconds,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_total_tokens=call_metrics.llm_total_tokens,
                tts_provider=tts_provider,
                tts_model=tts_model,
                tts_characters=call_metrics.tts_characters,
                total_cost_cents=int(call_metrics.total_cost_cents),
                avg_ttfa_ms=call_metrics.avg_ttfa_ms,
                avg_stt_latency_ms=call_metrics.avg_stt_ms,
                avg_llm_latency_ms=call_metrics.avg_llm_ms,
                avg_tts_latency_ms=call_metrics.avg_tts_ms,
            )
            self.db.add(usage)

    async def update_model_usage_stats(
        self,
        workspace_id: UUID,
        service_type: str,
        provider: str,
        model: str,
        request_count: int = 1,
        duration_ms: float = 0,
        audio_seconds: float = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        characters: int = 0,
        cost_cents: float = 0,
        error_count: int = 0,
    ) -> None:
        """Update model-level usage statistics."""
        today = date.today()

        # Use upsert
        stmt = insert(ModelUsageStats).values(
            workspace_id=str(workspace_id),
            date=today,
            service_type=service_type,
            provider=provider,
            model=model,
            request_count=request_count,
            total_duration_ms=duration_ms,
            total_audio_seconds=audio_seconds,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_characters=characters,
            total_cost_cents=int(cost_cents),
            error_count=error_count,
        ).on_conflict_do_update(
            constraint="uq_model_usage_stats_workspace_date_service",
            set_={
                "request_count": ModelUsageStats.request_count + request_count,
                "total_duration_ms": ModelUsageStats.total_duration_ms + duration_ms,
                "total_audio_seconds": ModelUsageStats.total_audio_seconds + audio_seconds,
                "total_input_tokens": ModelUsageStats.total_input_tokens + input_tokens,
                "total_output_tokens": ModelUsageStats.total_output_tokens + output_tokens,
                "total_characters": ModelUsageStats.total_characters + characters,
                "total_cost_cents": ModelUsageStats.total_cost_cents + int(cost_cents),
                "error_count": ModelUsageStats.error_count + error_count,
            }
        )

        await self.db.execute(stmt)

    async def log_error(
        self,
        service_type: str,
        error_message: str,
        workspace_id: Optional[UUID] = None,
        agent_id: Optional[UUID] = None,
        call_id: Optional[UUID] = None,
        message_id: Optional[UUID] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        error_code: Optional[str] = None,
        error_details: Optional[dict] = None,
        retry_count: int = 0,
        was_recovered: bool = False,
        stack_trace: Optional[str] = None,
    ) -> ErrorLog:
        """Log an error for debugging and tracking."""
        error = ErrorLog(
            workspace_id=str(workspace_id) if workspace_id else None,
            agent_id=str(agent_id) if agent_id else None,
            call_id=str(call_id) if call_id else None,
            message_id=str(message_id) if message_id else None,
            service_type=service_type,
            provider=provider,
            model=model,
            error_code=error_code,
            error_message=error_message,
            error_details=error_details,
            retry_count=retry_count,
            was_recovered=was_recovered,
            stack_trace=stack_trace,
        )

        self.db.add(error)
        await self.db.flush()

        return error

    async def create_prompt_version(
        self,
        agent_id: UUID,
        system_prompt: str,
        first_message: Optional[str] = None,
        created_by: Optional[UUID] = None,
        change_reason: Optional[str] = None,
    ) -> PromptVersion:
        """Create a new prompt version when agent prompt is updated."""
        # Get the next version number
        stmt = select(func.max(PromptVersion.version)).where(
            PromptVersion.agent_id == str(agent_id)
        )
        result = await self.db.execute(stmt)
        max_version = result.scalar() or 0

        # Deactivate current active version
        await self.db.execute(
            select(PromptVersion)
            .where(
                and_(
                    PromptVersion.agent_id == str(agent_id),
                    PromptVersion.is_active == True,
                )
            )
        )

        # Create new version
        version = PromptVersion(
            agent_id=str(agent_id),
            version=max_version + 1,
            system_prompt=system_prompt,
            first_message=first_message,
            created_by=str(created_by) if created_by else None,
            change_reason=change_reason,
            is_active=True,
        )

        self.db.add(version)
        await self.db.flush()

        return version

    async def get_agent_analytics(
        self,
        agent_id: UUID,
        start_date: date,
        end_date: date,
    ) -> List[AgentUsageDaily]:
        """Get daily usage records for an agent."""
        stmt = select(AgentUsageDaily).where(
            and_(
                AgentUsageDaily.agent_id == str(agent_id),
                AgentUsageDaily.date >= start_date,
                AgentUsageDaily.date <= end_date,
            )
        ).order_by(AgentUsageDaily.date)

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_workspace_model_stats(
        self,
        workspace_id: UUID,
        start_date: date,
        end_date: date,
    ) -> List[ModelUsageStats]:
        """Get model usage stats for a workspace."""
        stmt = select(ModelUsageStats).where(
            and_(
                ModelUsageStats.workspace_id == str(workspace_id),
                ModelUsageStats.date >= start_date,
                ModelUsageStats.date <= end_date,
            )
        ).order_by(ModelUsageStats.date)

        result = await self.db.execute(stmt)
        return result.scalars().all()
