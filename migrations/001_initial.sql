-- Voice Agent SaaS - Initial Schema
-- PostgreSQL 16

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users (basit, auth sonra eklenecek)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE NOT NULL,
    name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Voice Agents
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    -- Basic info
    name TEXT NOT NULL,
    description TEXT,

    -- Voice config
    system_prompt TEXT NOT NULL,
    language TEXT DEFAULT 'tr',
    voice_id TEXT DEFAULT 'leyla',
    voice_speed FLOAT DEFAULT 1.0,

    -- Provider config
    stt_provider TEXT DEFAULT 'deepgram',
    llm_provider TEXT DEFAULT 'openai',
    llm_model TEXT DEFAULT 'gpt-4o-mini-realtime-preview',
    tts_provider TEXT DEFAULT 'cartesia',

    -- Status
    is_active BOOLEAN DEFAULT true,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Voice Sessions (analytics)
CREATE TABLE IF NOT EXISTS voice_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,

    -- Session data
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    duration_seconds INT,

    -- Metrics
    message_count INT DEFAULT 0,
    total_tokens INT DEFAULT 0,
    stt_minutes FLOAT DEFAULT 0,
    tts_characters INT DEFAULT 0,

    -- Cost (cents)
    cost_cents INT DEFAULT 0,

    -- Metadata
    source TEXT DEFAULT 'web',
    metadata JSONB DEFAULT '{}'
);

-- Conversation messages
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES voice_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_agents_user_id ON agents(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_agent_id ON voice_sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);

-- Demo user (development)
INSERT INTO users (id, email, name)
VALUES ('00000000-0000-0000-0000-000000000001', 'demo@example.com', 'Demo User')
ON CONFLICT (email) DO NOTHING;

-- Demo agent
INSERT INTO agents (user_id, name, description, system_prompt)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'Asistan',
    'Genel amaçlı Türkçe asistan',
    'Sen yardımcı bir Türkçe asistansın. Kısa ve öz cevaplar ver.'
)
ON CONFLICT DO NOTHING;
