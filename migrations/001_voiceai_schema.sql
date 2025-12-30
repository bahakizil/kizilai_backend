-- =============================================================================
-- VoiceAI Database Schema for Supabase
-- =============================================================================
-- This schema extends Supabase's built-in auth.users table
-- Run this after Supabase is initialized

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- =============================================================================
-- Workspaces (Multi-tenant)
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.workspaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    slug TEXT UNIQUE NOT NULL,
    logo_url TEXT,
    plan TEXT DEFAULT 'free' CHECK (plan IN ('free', 'pro', 'business', 'enterprise')),
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Workspace Members (Links auth.users to workspaces)
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.workspace_members (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    role TEXT DEFAULT 'member' CHECK (role IN ('owner', 'admin', 'member')),
    invited_by UUID REFERENCES auth.users(id),
    invited_at TIMESTAMPTZ DEFAULT NOW(),
    joined_at TIMESTAMPTZ,
    UNIQUE(workspace_id, user_id)
);

-- =============================================================================
-- User Profiles (Extends auth.users)
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    full_name TEXT,
    avatar_url TEXT,
    phone TEXT,
    timezone TEXT DEFAULT 'Europe/Istanbul',
    language TEXT DEFAULT 'tr',
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Voice Agents
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,

    -- Basic Info
    name TEXT NOT NULL,
    description TEXT,
    avatar_url TEXT,
    status TEXT DEFAULT 'draft' CHECK (status IN ('draft', 'active', 'paused', 'archived')),
    is_active BOOLEAN DEFAULT true,

    -- Personality & Behavior
    system_prompt TEXT NOT NULL,
    first_message TEXT,
    language TEXT DEFAULT 'tr',

    -- Voice Configuration
    voice_provider TEXT DEFAULT 'cartesia',
    voice_id TEXT DEFAULT 'leyla',
    voice_speed FLOAT DEFAULT 1.0 CHECK (voice_speed >= 0.5 AND voice_speed <= 2.0),
    voice_pitch FLOAT DEFAULT 1.0 CHECK (voice_pitch >= 0.5 AND voice_pitch <= 2.0),
    voice_emotion TEXT DEFAULT 'neutral',

    -- STT Configuration
    stt_provider TEXT DEFAULT 'deepgram',
    stt_model TEXT DEFAULT 'nova-3',
    stt_language TEXT DEFAULT 'tr',

    -- LLM Configuration
    llm_provider TEXT DEFAULT 'openai',
    llm_model TEXT DEFAULT 'gpt-4o-mini',
    llm_temperature FLOAT DEFAULT 0.7 CHECK (llm_temperature >= 0 AND llm_temperature <= 2),
    llm_max_tokens INT DEFAULT 1024,

    -- TTS Configuration
    tts_provider TEXT DEFAULT 'cartesia',
    tts_model TEXT DEFAULT 'sonic-3',

    -- Call Settings
    max_duration_seconds INT DEFAULT 600,
    silence_timeout_seconds INT DEFAULT 30,
    interrupt_enabled BOOLEAN DEFAULT true,

    -- Stats (denormalized for performance)
    total_calls INT DEFAULT 0,
    total_minutes FLOAT DEFAULT 0,
    success_rate FLOAT DEFAULT 0,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Knowledge Bases (RAG)
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.knowledge_bases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES public.agents(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    chunk_size INT DEFAULT 500,
    chunk_overlap INT DEFAULT 50,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    knowledge_base_id UUID NOT NULL REFERENCES public.knowledge_bases(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    type TEXT CHECK (type IN ('pdf', 'docx', 'txt', 'md', 'url', 'manual')),
    source_url TEXT,
    content TEXT,
    file_size INT,
    chunk_count INT DEFAULT 0,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536), -- OpenAI ada-002 dimension
    token_count INT,
    chunk_index INT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Agent Functions (Tools)
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.agent_functions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES public.agents(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    parameters JSONB NOT NULL, -- JSON Schema
    webhook_url TEXT,
    webhook_method TEXT DEFAULT 'POST',
    webhook_headers JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Channels
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.phone_numbers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES public.agents(id) ON DELETE SET NULL,
    phone_number TEXT UNIQUE NOT NULL,
    twilio_sid TEXT,
    country TEXT DEFAULT 'TR',
    capabilities JSONB DEFAULT '{"voice": true, "sms": false}',
    is_active BOOLEAN DEFAULT true,
    monthly_cost_cents INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.web_widgets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID NOT NULL REFERENCES public.agents(id) ON DELETE CASCADE,
    name TEXT,
    button_color TEXT DEFAULT '#000000',
    button_icon TEXT DEFAULT 'microphone',
    button_position TEXT DEFAULT 'bottom-right',
    button_size TEXT DEFAULT 'medium',
    greeting_message TEXT,
    allowed_domains TEXT[],
    custom_css TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Calls & Messages
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.calls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES public.agents(id) ON DELETE SET NULL,

    -- Source
    channel TEXT NOT NULL CHECK (channel IN ('web', 'phone', 'api')),
    phone_number_id UUID REFERENCES public.phone_numbers(id),
    widget_id UUID REFERENCES public.web_widgets(id),

    -- Caller Info
    caller_id TEXT,
    caller_phone TEXT,
    caller_name TEXT,

    -- Timing
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    duration_seconds INT,

    -- Status
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'completed', 'failed', 'transferred', 'voicemail')),
    end_reason TEXT,

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
    user_rating INT CHECK (user_rating >= 1 AND user_rating <= 5),
    sentiment TEXT CHECK (sentiment IN ('positive', 'neutral', 'negative')),
    summary TEXT,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id UUID NOT NULL REFERENCES public.calls(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'function')),
    content TEXT,
    audio_url TEXT,

    -- Metrics
    stt_ms FLOAT,
    llm_ms FLOAT,
    tts_ms FLOAT,
    ttfa_ms FLOAT,

    -- Function Call
    function_name TEXT,
    function_args JSONB,
    function_result JSONB,

    -- RAG
    rag_chunks JSONB,

    -- Status
    was_interrupted BOOLEAN DEFAULT false,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- API Keys
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    key_hash TEXT UNIQUE NOT NULL,
    key_prefix TEXT NOT NULL, -- First 8 chars for display
    permissions JSONB DEFAULT '["*"]',
    rate_limit INT DEFAULT 1000,
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Webhooks
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.webhooks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    events TEXT[] NOT NULL,
    secret TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    retry_count INT DEFAULT 3,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.webhook_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    webhook_id UUID NOT NULL REFERENCES public.webhooks(id) ON DELETE CASCADE,
    event TEXT NOT NULL,
    payload JSONB,
    response_status INT,
    response_body TEXT,
    duration_ms INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- Usage & Billing
-- =============================================================================

CREATE TABLE IF NOT EXISTS public.usage_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    date DATE NOT NULL,

    -- Usage
    stt_seconds FLOAT DEFAULT 0,
    llm_tokens INT DEFAULT 0,
    tts_characters INT DEFAULT 0,
    phone_minutes FLOAT DEFAULT 0,
    api_calls INT DEFAULT 0,

    -- Cost
    cost_cents INT DEFAULT 0,

    UNIQUE(workspace_id, date)
);

CREATE TABLE IF NOT EXISTS public.invoices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    stripe_invoice_id TEXT UNIQUE,
    amount_cents INT NOT NULL,
    currency TEXT DEFAULT 'usd',
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'paid', 'failed', 'refunded')),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    pdf_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    paid_at TIMESTAMPTZ
);

-- =============================================================================
-- Indexes
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_workspace_members_user ON public.workspace_members(user_id);
CREATE INDEX IF NOT EXISTS idx_workspace_members_workspace ON public.workspace_members(workspace_id);
CREATE INDEX IF NOT EXISTS idx_agents_workspace ON public.agents(workspace_id);
CREATE INDEX IF NOT EXISTS idx_agents_status ON public.agents(status);
CREATE INDEX IF NOT EXISTS idx_calls_workspace ON public.calls(workspace_id);
CREATE INDEX IF NOT EXISTS idx_calls_agent ON public.calls(agent_id);
CREATE INDEX IF NOT EXISTS idx_calls_started ON public.calls(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_calls_status ON public.calls(status);
CREATE INDEX IF NOT EXISTS idx_messages_call ON public.messages(call_id);
CREATE INDEX IF NOT EXISTS idx_documents_knowledge_base ON public.documents(knowledge_base_id);
CREATE INDEX IF NOT EXISTS idx_usage_records_workspace_date ON public.usage_records(workspace_id, date);

-- Vector similarity search index
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON public.document_chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- =============================================================================
-- Functions & Triggers
-- =============================================================================

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION public.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at trigger to tables
CREATE TRIGGER update_workspaces_updated_at BEFORE UPDATE ON public.workspaces
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON public.agents
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON public.documents
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER update_knowledge_bases_updated_at BEFORE UPDATE ON public.knowledge_bases
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

-- Update agent stats after call
CREATE OR REPLACE FUNCTION public.update_agent_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'completed' AND NEW.agent_id IS NOT NULL THEN
        UPDATE public.agents SET
            total_calls = total_calls + 1,
            total_minutes = total_minutes + COALESCE(NEW.duration_seconds, 0) / 60.0,
            success_rate = (
                SELECT
                    CASE WHEN COUNT(*) > 0
                    THEN (COUNT(*) FILTER (WHERE status = 'completed')::FLOAT / COUNT(*)::FLOAT) * 100
                    ELSE 0 END
                FROM public.calls WHERE agent_id = NEW.agent_id
            )
        WHERE id = NEW.agent_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_agent_stats_on_call AFTER INSERT OR UPDATE ON public.calls
    FOR EACH ROW EXECUTE FUNCTION public.update_agent_stats();

-- Create profile on user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, full_name, avatar_url)
    VALUES (
        NEW.id,
        NEW.raw_user_meta_data->>'full_name',
        NEW.raw_user_meta_data->>'avatar_url'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- =============================================================================
-- Row Level Security (RLS)
-- =============================================================================

-- Enable RLS
ALTER TABLE public.workspaces ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.workspace_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.calls ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.knowledge_bases ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.webhooks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.usage_records ENABLE ROW LEVEL SECURITY;

-- Profiles: Users can only see/edit their own profile
CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

-- Workspaces: Members can view their workspaces
CREATE POLICY "Members can view workspaces" ON public.workspaces
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.workspace_members
            WHERE workspace_id = workspaces.id AND user_id = auth.uid()
        )
    );

-- Workspace members: Can see members of own workspaces
CREATE POLICY "Members can view workspace members" ON public.workspace_members
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.workspace_members wm
            WHERE wm.workspace_id = workspace_members.workspace_id AND wm.user_id = auth.uid()
        )
    );

-- Agents: Members can manage agents in their workspaces
CREATE POLICY "Members can view agents" ON public.agents
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.workspace_members
            WHERE workspace_id = agents.workspace_id AND user_id = auth.uid()
        )
    );

CREATE POLICY "Admins can manage agents" ON public.agents
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM public.workspace_members
            WHERE workspace_id = agents.workspace_id
            AND user_id = auth.uid()
            AND role IN ('owner', 'admin')
        )
    );

-- Calls: Members can view calls in their workspaces
CREATE POLICY "Members can view calls" ON public.calls
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.workspace_members
            WHERE workspace_id = calls.workspace_id AND user_id = auth.uid()
        )
    );

-- Messages: Members can view messages of calls in their workspaces
CREATE POLICY "Members can view messages" ON public.messages
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.calls c
            JOIN public.workspace_members wm ON wm.workspace_id = c.workspace_id
            WHERE c.id = messages.call_id AND wm.user_id = auth.uid()
        )
    );

-- =============================================================================
-- Demo Data
-- =============================================================================

-- Create demo workspace
INSERT INTO public.workspaces (id, name, slug, plan)
VALUES ('00000000-0000-0000-0000-000000000001', 'Demo Workspace', 'demo', 'pro')
ON CONFLICT (id) DO NOTHING;

-- Create demo agent
INSERT INTO public.agents (
    id, workspace_id, name, description, system_prompt, first_message, status
) VALUES (
    '00000000-0000-0000-0000-000000000002',
    '00000000-0000-0000-0000-000000000001',
    'Leyla - Müşteri Hizmetleri',
    'Türkçe konuşan profesyonel müşteri hizmetleri asistanı',
    'Sen Leyla, profesyonel ve yardımsever bir müşteri hizmetleri temsilcisisin. Türkçe konuşuyorsun ve müşterilere nazik, sabırlı bir şekilde yardımcı oluyorsun. Sorulara kısa ve öz cevaplar veriyorsun.',
    'Merhaba! Ben Leyla, size nasıl yardımcı olabilirim?',
    'active'
) ON CONFLICT (id) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO anon, authenticated;
