"""
VoiceAI - Modular Application Configuration
All settings are controlled via environment variables and Pydantic settings.
No hardcoded values.
"""
from functools import lru_cache
from typing import Optional, List, Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Base Settings
# =============================================================================

class AppSettings(BaseSettings):
    """Core application settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    name: str = Field(default="VoiceAI", alias="APP_NAME")
    version: str = Field(default="0.1.0", alias="APP_VERSION")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", alias="ENVIRONMENT"
    )
    debug: bool = Field(default=True, alias="DEBUG")
    secret_key: str = Field(
        default="change-me-in-production-with-secure-key",
        alias="SECRET_KEY"
    )

    # Server
    host: str = Field(default="0.0.0.0", alias="SERVER_HOST")
    port: int = Field(default=8000, alias="SERVER_PORT")
    workers: int = Field(default=1, alias="SERVER_WORKERS")
    reload: bool = Field(default=True, alias="SERVER_RELOAD")
    log_level: str = Field(default="info", alias="SERVER_LOG_LEVEL")

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        alias="CORS_ORIGINS"
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v


# =============================================================================
# Supabase Settings
# =============================================================================

class SupabaseSettings(BaseSettings):
    """Supabase connection settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    url: str = Field(default="http://localhost:8000", alias="SUPABASE_URL")
    anon_key: str = Field(
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyAgCiAgICAicm9sZSI6ICJhbm9uIiwKICAgICJpc3MiOiAic3VwYWJhc2UtZGVtbyIsCiAgICAiaWF0IjogMTY0MTc2OTIwMCwKICAgICJleHAiOiAxNzk5NTM1NjAwCn0.dc_X5iR_VP_qT0zsiyj_I_OZ2T9FtRU2BBNWN8Bu4GE",
        alias="SUPABASE_ANON_KEY"
    )
    service_role_key: str = Field(
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyAgCiAgICAicm9sZSI6ICJzZXJ2aWNlX3JvbGUiLAogICAgImlzcyI6ICJzdXBhYmFzZS1kZW1vIiwKICAgICJpYXQiOiAxNjQxNzY5MjAwLAogICAgImV4cCI6IDE3OTk1MzU2MDAKfQ.DaYlNEoUrrEn2Ig7tqibS-PHK5vgusbcbo7X36XVt4Q",
        alias="SUPABASE_SERVICE_ROLE_KEY"
    )
    jwt_secret: str = Field(
        default="voiceai-super-secret-jwt-token-with-at-least-32-characters-long-secret",
        alias="SUPABASE_JWT_SECRET"
    )

    # Database (Direct connection)
    db_host: str = Field(default="localhost", alias="SUPABASE_DB_HOST")
    db_port: int = Field(default=5435, alias="SUPABASE_DB_PORT")
    db_name: str = Field(default="postgres", alias="SUPABASE_DB_NAME")
    db_user: str = Field(default="postgres", alias="SUPABASE_DB_USER")
    db_password: str = Field(
        default="voiceai-super-secret-db-password-2024",
        alias="SUPABASE_DB_PASSWORD"
    )

    # Pooler connection (for production)
    pooler_port: int = Field(default=6544, alias="SUPABASE_POOLER_PORT")
    use_pooler: bool = Field(default=False, alias="SUPABASE_USE_POOLER")

    @property
    def database_url(self) -> str:
        """Get the async database URL."""
        port = self.pooler_port if self.use_pooler else self.db_port
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{port}/{self.db_name}"

    @property
    def sync_database_url(self) -> str:
        """Get the sync database URL for Alembic."""
        port = self.pooler_port if self.use_pooler else self.db_port
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{port}/{self.db_name}"


# =============================================================================
# Redis Settings
# =============================================================================

class RedisSettings(BaseSettings):
    """Redis connection settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    host: str = Field(default="localhost", alias="REDIS_HOST")
    port: int = Field(default=6379, alias="REDIS_PORT")
    db: int = Field(default=0, alias="REDIS_DB")
    password: Optional[str] = Field(default=None, alias="REDIS_PASSWORD")


# =============================================================================
# STT (Speech-to-Text) Provider Settings
# =============================================================================

class DeepgramSettings(BaseSettings):
    """Deepgram STT settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: Optional[str] = Field(default=None, alias="DEEPGRAM_API_KEY")
    model: str = Field(default="nova-3", alias="DEEPGRAM_MODEL")
    language: str = Field(default="tr", alias="DEEPGRAM_LANGUAGE")

    # Advanced options
    smart_format: bool = Field(default=True, alias="DEEPGRAM_SMART_FORMAT")
    punctuate: bool = Field(default=True, alias="DEEPGRAM_PUNCTUATE")
    diarize: bool = Field(default=False, alias="DEEPGRAM_DIARIZE")
    interim_results: bool = Field(default=True, alias="DEEPGRAM_INTERIM_RESULTS")
    endpointing: int = Field(default=300, alias="DEEPGRAM_ENDPOINTING")
    utterance_end_ms: int = Field(default=1000, alias="DEEPGRAM_UTTERANCE_END_MS")


class AssemblyAISettings(BaseSettings):
    """AssemblyAI STT settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: Optional[str] = Field(default=None, alias="ASSEMBLYAI_API_KEY")
    language: str = Field(default="tr", alias="ASSEMBLYAI_LANGUAGE")


class STTSettings(BaseSettings):
    """Combined STT settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    provider: Literal["deepgram", "assemblyai", "whisper"] = Field(
        default="deepgram", alias="STT_PROVIDER"
    )
    deepgram: DeepgramSettings = Field(default_factory=DeepgramSettings)
    assemblyai: AssemblyAISettings = Field(default_factory=AssemblyAISettings)


# =============================================================================
# TTS (Text-to-Speech) Provider Settings
# =============================================================================

class CartesiaSettings(BaseSettings):
    """Cartesia TTS settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: Optional[str] = Field(default=None, alias="CARTESIA_API_KEY")
    model: str = Field(default="sonic-3", alias="CARTESIA_MODEL")
    voice_id: str = Field(default="79a125e8-cd45-4c13-8a67-188112f4dd22", alias="CARTESIA_VOICE_ID")
    language: str = Field(default="tr", alias="CARTESIA_LANGUAGE")

    # Voice controls
    speed: float = Field(default=1.0, ge=0.5, le=2.0, alias="CARTESIA_SPEED")
    emotion: str = Field(default="neutral", alias="CARTESIA_EMOTION")


class ElevenLabsSettings(BaseSettings):
    """ElevenLabs TTS settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: Optional[str] = Field(default=None, alias="ELEVENLABS_API_KEY")
    voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM", alias="ELEVENLABS_VOICE_ID")
    model: str = Field(default="eleven_flash_v2_5", alias="ELEVENLABS_MODEL")
    stability: float = Field(default=0.5, ge=0.0, le=1.0, alias="ELEVENLABS_STABILITY")
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0, alias="ELEVENLABS_SIMILARITY")


class GoogleTTSSettings(BaseSettings):
    """Google Cloud TTS settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    credentials_path: Optional[str] = Field(default=None, alias="GOOGLE_APPLICATION_CREDENTIALS")
    language: str = Field(default="tr-TR", alias="GOOGLE_TTS_LANGUAGE")
    voice: str = Field(default="tr-TR-Wavenet-A", alias="GOOGLE_TTS_VOICE")
    speaking_rate: float = Field(default=1.0, ge=0.25, le=4.0, alias="GOOGLE_TTS_SPEAKING_RATE")
    pitch: float = Field(default=0.0, ge=-20.0, le=20.0, alias="GOOGLE_TTS_PITCH")


class TTSSettings(BaseSettings):
    """Combined TTS settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    provider: Literal["cartesia", "elevenlabs", "google", "openai", "azure"] = Field(
        default="cartesia", alias="TTS_PROVIDER"
    )
    cartesia: CartesiaSettings = Field(default_factory=CartesiaSettings)
    elevenlabs: ElevenLabsSettings = Field(default_factory=ElevenLabsSettings)
    google: GoogleTTSSettings = Field(default_factory=GoogleTTSSettings)


# =============================================================================
# LLM (Large Language Model) Provider Settings
# =============================================================================

class OpenAISettings(BaseSettings):
    """OpenAI settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    realtime_model: str = Field(default="gpt-4o-realtime-preview", alias="OPENAI_REALTIME_MODEL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, alias="OPENAI_TEMPERATURE")
    max_tokens: int = Field(default=1024, alias="OPENAI_MAX_TOKENS")

    # TTS settings (OpenAI TTS)
    tts_model: str = Field(default="tts-1", alias="OPENAI_TTS_MODEL")
    tts_voice: str = Field(default="alloy", alias="OPENAI_TTS_VOICE")


class AnthropicSettings(BaseSettings):
    """Anthropic Claude settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    model: str = Field(default="claude-3-haiku-20240307", alias="ANTHROPIC_MODEL")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, alias="ANTHROPIC_TEMPERATURE")
    max_tokens: int = Field(default=1024, alias="ANTHROPIC_MAX_TOKENS")


class GeminiSettings(BaseSettings):
    """Google Gemini settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    model: str = Field(default="gemini-2.0-flash-exp", alias="GEMINI_MODEL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, alias="GEMINI_TEMPERATURE")
    max_tokens: int = Field(default=1024, alias="GEMINI_MAX_TOKENS")


class LLMSettings(BaseSettings):
    """Combined LLM settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    provider: Literal["openai", "anthropic", "gemini"] = Field(
        default="openai", alias="LLM_PROVIDER"
    )
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)


# =============================================================================
# External Service Settings
# =============================================================================

class TwilioSettings(BaseSettings):
    """Twilio phone integration settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    account_sid: Optional[str] = Field(default=None, alias="TWILIO_ACCOUNT_SID")
    auth_token: Optional[str] = Field(default=None, alias="TWILIO_AUTH_TOKEN")
    phone_number: Optional[str] = Field(default=None, alias="TWILIO_PHONE_NUMBER")

    @property
    def is_configured(self) -> bool:
        return bool(self.account_sid and self.auth_token)


class StripeSettings(BaseSettings):
    """Stripe billing settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    secret_key: Optional[str] = Field(default=None, alias="STRIPE_SECRET_KEY")
    publishable_key: Optional[str] = Field(default=None, alias="STRIPE_PUBLISHABLE_KEY")
    webhook_secret: Optional[str] = Field(default=None, alias="STRIPE_WEBHOOK_SECRET")

    # Price IDs for plans
    price_id_pro: Optional[str] = Field(default=None, alias="STRIPE_PRICE_ID_PRO")
    price_id_business: Optional[str] = Field(default=None, alias="STRIPE_PRICE_ID_BUSINESS")

    @property
    def is_configured(self) -> bool:
        return bool(self.secret_key)


# =============================================================================
# Recording Settings
# =============================================================================

class RecordingSettings(BaseSettings):
    """Audio recording configuration."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    enabled: bool = Field(default=True, alias="RECORDING_ENABLED")
    storage_provider: Literal["supabase", "s3", "gcs", "local"] = Field(
        default="supabase", alias="RECORDING_STORAGE_PROVIDER"
    )
    bucket: str = Field(default="call-recordings", alias="RECORDING_BUCKET")
    retention_days: int = Field(default=90, alias="RECORDING_RETENTION_DAYS")
    format: Literal["wav", "mp3", "opus"] = Field(default="wav", alias="RECORDING_FORMAT")
    sample_rate: int = Field(default=16000, alias="RECORDING_SAMPLE_RATE")
    channels: int = Field(default=1, alias="RECORDING_CHANNELS")

    # S3 configuration (if using S3)
    s3_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    s3_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    s3_bucket: Optional[str] = Field(default=None, alias="AWS_S3_BUCKET")
    s3_region: str = Field(default="eu-central-1", alias="AWS_S3_REGION")

    # Local storage path (for development)
    local_path: str = Field(default="./recordings", alias="RECORDING_LOCAL_PATH")

    @property
    def is_s3_configured(self) -> bool:
        return bool(self.s3_access_key_id and self.s3_secret_access_key and self.s3_bucket)


# =============================================================================
# Voice Pipeline Settings
# =============================================================================

class PipelineSettings(BaseSettings):
    """Voice pipeline configuration."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Default providers (can be overridden per-agent)
    stt_provider: str = Field(default="deepgram", alias="PIPELINE_STT_PROVIDER")
    tts_provider: str = Field(default="cartesia", alias="PIPELINE_TTS_PROVIDER")
    llm_provider: str = Field(default="openai", alias="PIPELINE_LLM_PROVIDER")

    # Features
    enable_barge_in: bool = Field(default=True, alias="PIPELINE_ENABLE_BARGE_IN")
    enable_thinking_filler: bool = Field(default=True, alias="PIPELINE_ENABLE_THINKING_FILLER")
    enable_metrics: bool = Field(default=True, alias="PIPELINE_ENABLE_METRICS")
    enable_rag: bool = Field(default=True, alias="PIPELINE_ENABLE_RAG")

    # Timeouts & Thresholds
    thinking_threshold_ms: int = Field(default=800, alias="PIPELINE_THINKING_THRESHOLD_MS")
    silence_timeout_ms: int = Field(default=3000, alias="PIPELINE_SILENCE_TIMEOUT_MS")
    max_call_duration_seconds: int = Field(default=600, alias="PIPELINE_MAX_CALL_DURATION")

    # Audio settings
    sample_rate: int = Field(default=16000, alias="PIPELINE_SAMPLE_RATE")
    channels: int = Field(default=1, alias="PIPELINE_CHANNELS")
    chunk_size: int = Field(default=4096, alias="PIPELINE_CHUNK_SIZE")


# =============================================================================
# Main Settings Class
# =============================================================================

class Settings(BaseSettings):
    """
    Main application settings container.
    All configuration is loaded from environment variables.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Sub-configurations
    app: AppSettings = Field(default_factory=AppSettings)
    supabase: SupabaseSettings = Field(default_factory=SupabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    stt: STTSettings = Field(default_factory=STTSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    twilio: TwilioSettings = Field(default_factory=TwilioSettings)
    stripe: StripeSettings = Field(default_factory=StripeSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    recording: RecordingSettings = Field(default_factory=RecordingSettings)

    # Convenience properties
    @property
    def is_production(self) -> bool:
        return self.app.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.app.environment == "development"

    @property
    def database_url(self) -> str:
        return self.supabase.database_url

    def get_active_stt_config(self):
        """Get the active STT provider configuration."""
        provider = self.stt.provider
        if provider == "deepgram":
            return self.stt.deepgram
        elif provider == "assemblyai":
            return self.stt.assemblyai
        raise ValueError(f"Unknown STT provider: {provider}")

    def get_active_tts_config(self):
        """Get the active TTS provider configuration."""
        provider = self.tts.provider
        if provider == "cartesia":
            return self.tts.cartesia
        elif provider == "elevenlabs":
            return self.tts.elevenlabs
        elif provider == "google":
            return self.tts.google
        raise ValueError(f"Unknown TTS provider: {provider}")

    def get_active_llm_config(self):
        """Get the active LLM provider configuration."""
        provider = self.llm.provider
        if provider == "openai":
            return self.llm.openai
        elif provider == "anthropic":
            return self.llm.anthropic
        elif provider == "gemini":
            return self.llm.gemini
        raise ValueError(f"Unknown LLM provider: {provider}")


# =============================================================================
# Settings Instance
# =============================================================================

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
