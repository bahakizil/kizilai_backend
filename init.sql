-- VoiceAI Database Schema
-- PostgreSQL 16 with pgvector extension

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ============================================
-- USERS & AUTHENTICATION
-- ============================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT,
    name TEXT,
    avatar_url TEXT,
    email_verified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    provider TEXT NOT NULL, -- google, github, credentials
    provider_account_id TEXT,
    access_token TEXT,
    refresh_token TEXT,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(provider, provider_account_id)
);

CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token TEXT UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- MULTI-TENANT WORKSPACES
-- ============================================

CREATE TABLE workspaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    logo_url TEXT,
    plan TEXT DEFAULT 'free', -- free, pro, enterprise
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    timezone TEXT DEFAULT 'UTC',
    default_language TEXT DEFAULT 'tr',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE workspace_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role TEXT DEFAULT 'member', -- owner, admin, member
    invited_at TIMESTAMPTZ DEFAULT NOW(),
    joined_at TIMESTAMPTZ,
    UNIQUE(workspace_id, user_id)
);

CREATE TABLE workspace_invitations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    role TEXT DEFAULT 'member',
    token TEXT UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- VOICE AGENTS
-- ============================================

CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,

    -- Basic Info
    name TEXT NOT NULL,
    description TEXT,
    avatar_url TEXT,
    is_active BOOLEAN DEFAULT true,
    status TEXT DEFAULT 'draft', -- draft, active, paused

    -- Personality
    system_prompt TEXT NOT NULL,
    first_message TEXT,
    language TEXT DEFAULT 'tr',

    -- Voice Configuration
    voice_provider TEXT DEFAULT 'cartesia',
    voice_id TEXT DEFAULT 'leyla',
    voice_speed FLOAT DEFAULT 1.0,
    voice_pitch FLOAT DEFAULT 1.0,
    voice_emotion TEXT DEFAULT 'neutral',

    -- STT Configuration
    stt_provider TEXT DEFAULT 'deepgram',
    stt_model TEXT DEFAULT 'nova-3',
    stt_language TEXT DEFAULT 'tr',

    -- LLM Configuration
    llm_provider TEXT DEFAULT 'openai',
    llm_model TEXT DEFAULT 'gpt-4o-mini',
    llm_temperature FLOAT DEFAULT 0.7,
    llm_max_tokens INT DEFAULT 1024,

    -- TTS Configuration
    tts_provider TEXT DEFAULT 'cartesia',
    tts_model TEXT DEFAULT 'sonic-3',

    -- Advanced Settings
    max_duration_seconds INT DEFAULT 600,
    silence_timeout_seconds INT DEFAULT 30,
    interrupt_enabled BOOLEAN DEFAULT true,

    -- Denormalized Stats (for performance)
    total_calls INT DEFAULT 0,
    total_minutes FLOAT DEFAULT 0,
    avg_duration_seconds FLOAT DEFAULT 0,
    success_rate FLOAT DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- KNOWLEDGE BASE (RAG)
-- ============================================

CREATE TABLE knowledge_bases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    knowledge_base_id UUID REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    type TEXT, -- pdf, docx, txt, url, manual
    source_url TEXT,
    file_path TEXT,
    file_size_bytes INT,
    content TEXT,
    chunk_count INT DEFAULT 0,
    status TEXT DEFAULT 'pending', -- pending, processing, ready, failed
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(384), -- pgvector with all-MiniLM-L6-v2
    token_count INT,
    chunk_index INT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- FUNCTIONS & TOOLS
-- ============================================

CREATE TABLE agent_functions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    parameters JSONB NOT NULL, -- JSON Schema
    webhook_url TEXT,
    webhook_method TEXT DEFAULT 'POST',
    webhook_headers JSONB DEFAULT '{}',
    timeout_seconds INT DEFAULT 30,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE function_templates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT, -- calendar, crm, database, etc.
    parameters JSONB NOT NULL,
    is_public BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- CHANNELS
-- ============================================

CREATE TABLE phone_numbers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    phone_number TEXT UNIQUE NOT NULL,
    twilio_sid TEXT,
    country TEXT DEFAULT 'TR',
    capabilities JSONB DEFAULT '{"voice": true, "sms": false}',
    monthly_cost_cents INT DEFAULT 1500,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE web_widgets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    name TEXT,
    button_color TEXT DEFAULT '#7C3AED',
    button_icon TEXT DEFAULT 'microphone',
    button_position TEXT DEFAULT 'bottom-right',
    button_size TEXT DEFAULT 'medium',
    greeting_message TEXT,
    allowed_domains TEXT[], -- CORS
    custom_css TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- CONVERSATIONS & CALLS
-- ============================================

CREATE TABLE calls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents(id) ON DELETE SET NULL,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,

    -- Source
    channel TEXT NOT NULL, -- web, phone, api
    phone_number_id UUID REFERENCES phone_numbers(id),
    widget_id UUID REFERENCES web_widgets(id),

    -- Caller Info
    caller_id TEXT,
    caller_phone TEXT,
    caller_name TEXT,

    -- Timing
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    duration_seconds INT,

    -- Status
    status TEXT DEFAULT 'active', -- active, completed, failed, transferred
    end_reason TEXT, -- user_hangup, agent_hangup, timeout, error, transfer

    -- Metrics
    message_count INT DEFAULT 0,
    stt_seconds FLOAT DEFAULT 0,
    llm_tokens INT DEFAULT 0,
    tts_characters INT DEFAULT 0,
    avg_ttfa_ms FLOAT,
    avg_llm_ms FLOAT,
    avg_tts_ms FLOAT,

    -- Cost
    cost_cents INT DEFAULT 0,

    -- Quality
    user_rating INT, -- 1-5
    sentiment TEXT, -- positive, neutral, negative
    sentiment_score FLOAT,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    tags TEXT[],

    -- Summary (generated by LLM)
    summary TEXT,
    key_points JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id UUID REFERENCES calls(id) ON DELETE CASCADE,

    role TEXT NOT NULL, -- user, assistant, system, function
    content TEXT,
    audio_url TEXT,

    -- Latency Metrics
    stt_ms FLOAT,
    llm_ms FLOAT,
    tts_ms FLOAT,
    ttfa_ms FLOAT, -- Time to First Audio

    -- Function Call
    function_name TEXT,
    function_args JSONB,
    function_result JSONB,

    -- RAG Context
    rag_chunks JSONB, -- retrieved chunks with scores
    rag_query TEXT,

    -- Interruption
    was_interrupted BOOLEAN DEFAULT false,
    interrupted_at_ms INT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- API KEYS
-- ============================================

CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    key_hash TEXT UNIQUE NOT NULL, -- hashed with SHA-256
    key_prefix TEXT NOT NULL, -- first 8 chars for display
    permissions JSONB DEFAULT '["*"]',
    rate_limit_per_minute INT DEFAULT 1000,
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE api_key_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_key_id UUID REFERENCES api_keys(id) ON DELETE CASCADE,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL,
    status_code INT,
    response_time_ms INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- WEBHOOKS
-- ============================================

CREATE TABLE webhooks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    name TEXT,
    url TEXT NOT NULL,
    events TEXT[] NOT NULL, -- call.started, call.ended, message.created, etc.
    secret TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    retry_count INT DEFAULT 3,
    timeout_seconds INT DEFAULT 30,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE webhook_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    webhook_id UUID REFERENCES webhooks(id) ON DELETE CASCADE,
    event TEXT NOT NULL,
    payload JSONB,
    request_headers JSONB,
    response_status INT,
    response_body TEXT,
    response_time_ms INT,
    attempt_number INT DEFAULT 1,
    success BOOLEAN,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- USAGE & BILLING
-- ============================================

CREATE TABLE usage_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    date DATE NOT NULL,

    -- Voice Usage
    stt_seconds FLOAT DEFAULT 0,
    tts_characters INT DEFAULT 0,
    phone_minutes FLOAT DEFAULT 0,

    -- LLM Usage
    llm_input_tokens INT DEFAULT 0,
    llm_output_tokens INT DEFAULT 0,

    -- Counts
    call_count INT DEFAULT 0,
    message_count INT DEFAULT 0,
    api_request_count INT DEFAULT 0,

    -- Costs (in cents)
    stt_cost_cents INT DEFAULT 0,
    tts_cost_cents INT DEFAULT 0,
    llm_cost_cents INT DEFAULT 0,
    phone_cost_cents INT DEFAULT 0,
    total_cost_cents INT DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(workspace_id, date)
);

CREATE TABLE invoices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    stripe_invoice_id TEXT,
    amount_cents INT NOT NULL,
    currency TEXT DEFAULT 'usd',
    status TEXT DEFAULT 'draft', -- draft, open, paid, void
    period_start DATE,
    period_end DATE,
    pdf_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- INDEXES
-- ============================================

-- Users
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_accounts_user ON accounts(user_id);

-- Workspaces
CREATE INDEX idx_workspaces_slug ON workspaces(slug);
CREATE INDEX idx_workspace_members_user ON workspace_members(user_id);
CREATE INDEX idx_workspace_members_workspace ON workspace_members(workspace_id);

-- Agents
CREATE INDEX idx_agents_workspace ON agents(workspace_id);
CREATE INDEX idx_agents_status ON agents(workspace_id, status);

-- Knowledge Base
CREATE INDEX idx_knowledge_bases_agent ON knowledge_bases(agent_id);
CREATE INDEX idx_documents_knowledge_base ON documents(knowledge_base_id);
CREATE INDEX idx_document_chunks_document ON document_chunks(document_id);
CREATE INDEX idx_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Functions
CREATE INDEX idx_agent_functions_agent ON agent_functions(agent_id);

-- Channels
CREATE INDEX idx_phone_numbers_workspace ON phone_numbers(workspace_id);
CREATE INDEX idx_phone_numbers_agent ON phone_numbers(agent_id);
CREATE INDEX idx_web_widgets_agent ON web_widgets(agent_id);

-- Calls & Messages
CREATE INDEX idx_calls_workspace ON calls(workspace_id);
CREATE INDEX idx_calls_agent ON calls(agent_id);
CREATE INDEX idx_calls_started ON calls(started_at DESC);
CREATE INDEX idx_calls_status ON calls(workspace_id, status);
CREATE INDEX idx_messages_call ON messages(call_id);
CREATE INDEX idx_messages_created ON messages(call_id, created_at);

-- API Keys
CREATE INDEX idx_api_keys_workspace ON api_keys(workspace_id);
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);

-- Webhooks
CREATE INDEX idx_webhooks_workspace ON webhooks(workspace_id);
CREATE INDEX idx_webhook_logs_webhook ON webhook_logs(webhook_id);

-- Usage
CREATE INDEX idx_usage_records_workspace_date ON usage_records(workspace_id, date DESC);
CREATE INDEX idx_invoices_workspace ON invoices(workspace_id);

-- ============================================
-- DEMO DATA
-- ============================================

-- Create demo user
INSERT INTO users (id, email, password_hash, name, email_verified_at)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'demo@voiceai.app',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.S3n3lxfGYMYr1y', -- password: demo123
    'Demo User',
    NOW()
);

-- Create demo workspace
INSERT INTO workspaces (id, name, slug, plan)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'Demo Workspace',
    'demo',
    'pro'
);

-- Add demo user to workspace as owner
INSERT INTO workspace_members (workspace_id, user_id, role, joined_at)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000001',
    'owner',
    NOW()
);

-- Create demo agents
INSERT INTO agents (id, workspace_id, name, description, system_prompt, first_message, status, language, voice_id)
VALUES
(
    '00000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000001',
    'Customer Support',
    'Handles customer inquiries, complaints, and general support requests with empathy and efficiency.',
    'Sen TechStore''un müşteri hizmetleri temsilcisisin. Müşterilere nazik ve profesyonel bir şekilde yardım et. Sipariş durumu, iade ve değişim işlemleri hakkında bilgi ver. Çözemediğin sorunlar için özür dile ve insan temsilciye yönlendir.',
    'Merhaba! TechStore müşteri hizmetlerine hoş geldiniz. Size nasıl yardımcı olabilirim?',
    'active',
    'tr',
    'leyla'
),
(
    '00000000-0000-0000-0000-000000000002',
    '00000000-0000-0000-0000-000000000001',
    'Sales Assistant',
    'Qualifies leads, answers product questions, and schedules demos with potential customers.',
    'Sen TechStore''un satış asistanısın. Potansiyel müşterilere ürünler hakkında bilgi ver, ihtiyaçlarını anla ve uygun ürünleri öner. Demo randevuları ayarla.',
    'Merhaba! TechStore''a hoş geldiniz. Ürünlerimiz hakkında size nasıl yardımcı olabilirim?',
    'active',
    'tr',
    'ahmet'
),
(
    '00000000-0000-0000-0000-000000000003',
    '00000000-0000-0000-0000-000000000001',
    'Appointment Booking',
    'Schedules, reschedules, and cancels appointments. Integrates with calendar systems.',
    'Sen bir randevu asistanısın. Müşterilerin randevu almasına, değiştirmesine ve iptal etmesine yardım et. Takvimi kontrol et ve uygun saatleri öner.',
    'Merhaba! Randevu hizmetine hoş geldiniz. Nasıl yardımcı olabilirim?',
    'active',
    'tr',
    'leyla'
);

-- Create demo phone number
INSERT INTO phone_numbers (workspace_id, agent_id, phone_number, country, is_active)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    '00000000-0000-0000-0000-000000000001',
    '+90 212 555 0001',
    'TR',
    true
);

-- Create demo API key
INSERT INTO api_keys (workspace_id, name, key_hash, key_prefix, permissions)
VALUES (
    '00000000-0000-0000-0000-000000000001',
    'Development Key',
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', -- sha256 of empty string, replace in production
    'va_dev_',
    '["*"]'
);

-- ============================================
-- FUNCTIONS (Triggers)
-- ============================================

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_workspaces_updated_at BEFORE UPDATE ON workspaces FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_knowledge_bases_updated_at BEFORE UPDATE ON knowledge_bases FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_agent_functions_updated_at BEFORE UPDATE ON agent_functions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_phone_numbers_updated_at BEFORE UPDATE ON phone_numbers FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_web_widgets_updated_at BEFORE UPDATE ON web_widgets FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_webhooks_updated_at BEFORE UPDATE ON webhooks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update agent stats after call
CREATE OR REPLACE FUNCTION update_agent_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' OR (TG_OP = 'UPDATE' AND NEW.status = 'completed' AND OLD.status != 'completed') THEN
        UPDATE agents SET
            total_calls = (SELECT COUNT(*) FROM calls WHERE agent_id = NEW.agent_id AND status = 'completed'),
            total_minutes = (SELECT COALESCE(SUM(duration_seconds), 0) / 60.0 FROM calls WHERE agent_id = NEW.agent_id AND status = 'completed'),
            avg_duration_seconds = (SELECT COALESCE(AVG(duration_seconds), 0) FROM calls WHERE agent_id = NEW.agent_id AND status = 'completed'),
            success_rate = (SELECT COALESCE(
                COUNT(*) FILTER (WHERE status = 'completed') * 100.0 / NULLIF(COUNT(*), 0),
                0
            ) FROM calls WHERE agent_id = NEW.agent_id)
        WHERE id = NEW.agent_id;
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_agent_stats_trigger AFTER INSERT OR UPDATE ON calls FOR EACH ROW EXECUTE FUNCTION update_agent_stats();

COMMENT ON TABLE agents IS 'Voice AI agents with personality, voice, and model configurations';
COMMENT ON TABLE calls IS 'Voice call sessions between users and agents';
COMMENT ON TABLE messages IS 'Individual messages within a call session';
COMMENT ON TABLE document_chunks IS 'Vector-indexed chunks for RAG retrieval';
