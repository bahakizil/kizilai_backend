"""
Agent Endpoints
"""
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.models import Agent

router = APIRouter()

# Demo workspace ID
DEMO_WORKSPACE_ID = "00000000-0000-0000-0000-000000000001"


class AgentResponse(BaseModel):
    id: UUID
    workspace_id: UUID
    name: str
    description: Optional[str]
    status: str
    language: str
    voice_id: str
    voice_speed: float
    llm_model: str
    total_calls: int
    total_minutes: float
    success_rate: float

    class Config:
        from_attributes = True


class AgentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    system_prompt: str = Field(..., min_length=1)
    first_message: Optional[str] = None
    language: str = "tr"
    voice_id: str = "leyla"
    voice_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    voice_pitch: float = Field(default=1.0, ge=0.5, le=2.0)
    voice_emotion: str = "neutral"
    stt_provider: str = "deepgram"
    stt_model: str = "nova-3"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.7, ge=0, le=1)
    tts_provider: str = "cartesia"
    tts_model: str = "sonic-3"


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    first_message: Optional[str] = None
    language: Optional[str] = None
    status: Optional[str] = None
    voice_id: Optional[str] = None
    voice_speed: Optional[float] = None
    voice_pitch: Optional[float] = None
    voice_emotion: Optional[str] = None
    stt_provider: Optional[str] = None
    stt_model: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_temperature: Optional[float] = None
    tts_provider: Optional[str] = None
    tts_model: Optional[str] = None
    is_active: Optional[bool] = None


class AgentDetailResponse(AgentResponse):
    system_prompt: str
    first_message: Optional[str]
    voice_pitch: float
    voice_emotion: str
    voice_provider: str
    stt_provider: str
    stt_model: str
    llm_provider: str
    llm_temperature: float
    tts_provider: str
    tts_model: str
    is_active: bool
    max_duration_seconds: int
    silence_timeout_seconds: int
    interrupt_enabled: bool


@router.get("", response_model=List[AgentResponse])
async def list_agents(db: AsyncSession = Depends(get_db)):
    """List all agents in workspace."""
    result = await db.execute(
        select(Agent).where(Agent.workspace_id == DEMO_WORKSPACE_ID)
    )
    agents = result.scalars().all()
    return agents


@router.get("/{agent_id}", response_model=AgentDetailResponse)
async def get_agent(agent_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get agent by ID."""
    result = await db.execute(select(Agent).where(Agent.id == str(agent_id)))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.post("", response_model=AgentDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(data: AgentCreate, db: AsyncSession = Depends(get_db)):
    """Create new agent."""
    agent = Agent(
        workspace_id=DEMO_WORKSPACE_ID,
        **data.model_dump(),
    )
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    return agent


@router.patch("/{agent_id}", response_model=AgentDetailResponse)
async def update_agent(
    agent_id: UUID,
    data: AgentUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update agent."""
    result = await db.execute(select(Agent).where(Agent.id == str(agent_id)))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(agent, field, value)

    await db.commit()
    await db.refresh(agent)
    return agent


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(agent_id: UUID, db: AsyncSession = Depends(get_db)):
    """Delete agent."""
    result = await db.execute(select(Agent).where(Agent.id == str(agent_id)))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    await db.delete(agent)
    await db.commit()


@router.post("/{agent_id}/duplicate", response_model=AgentDetailResponse)
async def duplicate_agent(agent_id: UUID, db: AsyncSession = Depends(get_db)):
    """Duplicate an agent."""
    result = await db.execute(select(Agent).where(Agent.id == str(agent_id)))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Create a copy
    new_agent = Agent(
        workspace_id=agent.workspace_id,
        name=f"{agent.name} (Copy)",
        description=agent.description,
        system_prompt=agent.system_prompt,
        first_message=agent.first_message,
        language=agent.language,
        voice_provider=agent.voice_provider,
        voice_id=agent.voice_id,
        voice_speed=agent.voice_speed,
        voice_pitch=agent.voice_pitch,
        voice_emotion=agent.voice_emotion,
        stt_provider=agent.stt_provider,
        stt_model=agent.stt_model,
        llm_provider=agent.llm_provider,
        llm_model=agent.llm_model,
        llm_temperature=agent.llm_temperature,
        tts_provider=agent.tts_provider,
        tts_model=agent.tts_model,
        status="draft",
    )
    db.add(new_agent)
    await db.commit()
    await db.refresh(new_agent)
    return new_agent
