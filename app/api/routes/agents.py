"""
Agent CRUD Routes
"""
from fastapi import APIRouter, HTTPException
from typing import List
from uuid import UUID
from ...models.agent import Agent, AgentCreate, AgentUpdate
from ...database import fetch_all, fetch_one, execute

router = APIRouter()

# Demo user ID (auth olmadan)
DEMO_USER_ID = "00000000-0000-0000-0000-000000000001"


@router.get("", response_model=List[Agent])
async def list_agents():
    """List all agents for current user."""
    rows = await fetch_all(
        """
        SELECT id, user_id, name, description, system_prompt, language,
               voice_id, voice_speed, stt_provider, llm_provider, llm_model,
               tts_provider, is_active, created_at, updated_at
        FROM agents
        WHERE user_id = $1
        ORDER BY created_at DESC
        """,
        DEMO_USER_ID
    )
    return [dict(row) for row in rows]


@router.post("", response_model=Agent)
async def create_agent(agent: AgentCreate):
    """Create a new agent."""
    row = await fetch_one(
        """
        INSERT INTO agents (user_id, name, description, system_prompt, language,
                           voice_id, voice_speed, stt_provider, llm_provider,
                           llm_model, tts_provider)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING id, user_id, name, description, system_prompt, language,
                  voice_id, voice_speed, stt_provider, llm_provider, llm_model,
                  tts_provider, is_active, created_at, updated_at
        """,
        DEMO_USER_ID,
        agent.name,
        agent.description,
        agent.system_prompt,
        agent.language,
        agent.voice_id,
        agent.voice_speed,
        agent.stt_provider,
        agent.llm_provider,
        agent.llm_model,
        agent.tts_provider
    )
    return dict(row)


@router.get("/{agent_id}", response_model=Agent)
async def get_agent(agent_id: UUID):
    """Get agent by ID."""
    row = await fetch_one(
        """
        SELECT id, user_id, name, description, system_prompt, language,
               voice_id, voice_speed, stt_provider, llm_provider, llm_model,
               tts_provider, is_active, created_at, updated_at
        FROM agents
        WHERE id = $1 AND user_id = $2
        """,
        str(agent_id),
        DEMO_USER_ID
    )
    if not row:
        raise HTTPException(status_code=404, detail="Agent not found")
    return dict(row)


@router.put("/{agent_id}", response_model=Agent)
async def update_agent(agent_id: UUID, agent: AgentUpdate):
    """Update agent."""
    # Get existing agent
    existing = await fetch_one(
        "SELECT * FROM agents WHERE id = $1 AND user_id = $2",
        str(agent_id),
        DEMO_USER_ID
    )
    if not existing:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Build update query
    updates = agent.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    set_clauses = []
    values = []
    for i, (key, value) in enumerate(updates.items(), start=1):
        set_clauses.append(f"{key} = ${i}")
        values.append(value)

    values.append(str(agent_id))
    values.append(DEMO_USER_ID)

    query = f"""
        UPDATE agents
        SET {", ".join(set_clauses)}, updated_at = NOW()
        WHERE id = ${len(values) - 1} AND user_id = ${len(values)}
        RETURNING id, user_id, name, description, system_prompt, language,
                  voice_id, voice_speed, stt_provider, llm_provider, llm_model,
                  tts_provider, is_active, created_at, updated_at
    """

    row = await fetch_one(query, *values)
    return dict(row)


@router.delete("/{agent_id}")
async def delete_agent(agent_id: UUID):
    """Delete agent."""
    result = await execute(
        "DELETE FROM agents WHERE id = $1 AND user_id = $2",
        str(agent_id),
        DEMO_USER_ID
    )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": "Agent deleted"}
