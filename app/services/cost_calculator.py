"""
Cost Calculator Service
Calculate costs for STT, LLM, and TTS services based on provider pricing.
"""
from typing import Optional
from dataclasses import dataclass


@dataclass
class CostBreakdown:
    """Cost breakdown for a message or call."""
    stt_cost_cents: float = 0.0
    llm_cost_cents: float = 0.0
    tts_cost_cents: float = 0.0
    total_cost_cents: float = 0.0


class CostCalculator:
    """Calculate costs for various AI services."""

    # Provider pricing (in cents)
    # Updated as of December 2024
    PRICING = {
        "stt": {
            "deepgram": {
                # Nova-3: $0.0043/second for Turkish
                "nova-3": 0.0043,
                "nova-2": 0.0036,
                "base": 0.0025,
                # Whisper through Deepgram
                "whisper-large": 0.0048,
            },
            "openai": {
                # Whisper: $0.006/minute = $0.0001/second
                "whisper-1": 0.01,  # per second
            },
            "google": {
                # Google STT: $0.006/15 seconds = $0.0004/second
                "latest_long": 0.0004,
                "latest_short": 0.0004,
            },
        },
        "llm": {
            # Pricing per 1K tokens
            "openai": {
                "gpt-4o": {"input": 0.25, "output": 1.0},  # $2.50/$10.00 per 1M
                "gpt-4o-mini": {"input": 0.015, "output": 0.06},  # $0.15/$0.60 per 1M
                "gpt-4o-realtime": {"input": 0.5, "output": 2.0},  # Realtime API pricing
                "gpt-4-turbo": {"input": 1.0, "output": 3.0},
                "gpt-3.5-turbo": {"input": 0.05, "output": 0.15},
            },
            "anthropic": {
                "claude-3-opus": {"input": 1.5, "output": 7.5},
                "claude-3-sonnet": {"input": 0.3, "output": 1.5},
                "claude-3-haiku": {"input": 0.025, "output": 0.125},
                "claude-3.5-sonnet": {"input": 0.3, "output": 1.5},
            },
            "google": {
                "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free tier
                "gemini-1.5-pro": {"input": 0.125, "output": 0.5},
                "gemini-1.5-flash": {"input": 0.0075, "output": 0.03},
            },
        },
        "tts": {
            # Pricing per 1K characters
            "cartesia": {
                # Cartesia Sonic: $0.15/1K chars
                "sonic-3": 0.015,
                "sonic-turbo": 0.015,
                "sonic-english": 0.015,
            },
            "elevenlabs": {
                # ElevenLabs: varies by tier, ~$0.30/1K chars
                "eleven_turbo_v2_5": 0.03,
                "eleven_turbo_v2": 0.03,
                "eleven_multilingual_v2": 0.03,
                "eleven_flash_v2_5": 0.02,
            },
            "openai": {
                # OpenAI TTS: $15/1M chars = $0.015/1K chars
                "tts-1": 0.0015,
                "tts-1-hd": 0.003,
            },
            "google": {
                # Google TTS: $4/1M chars = $0.004/1K chars (Wavenet)
                "wavenet": 0.0004,
                "neural2": 0.0006,
                "standard": 0.0002,
            },
        },
    }

    @classmethod
    def calculate_stt_cost(
        cls,
        provider: str,
        model: str,
        audio_seconds: float,
    ) -> float:
        """
        Calculate STT cost in cents.

        Args:
            provider: STT provider (deepgram, openai, google)
            model: Model name
            audio_seconds: Duration of audio in seconds

        Returns:
            Cost in cents
        """
        provider = provider.lower()
        model = model.lower()

        provider_pricing = cls.PRICING["stt"].get(provider, {})
        rate_per_second = provider_pricing.get(model, 0.0)

        if rate_per_second == 0 and provider_pricing:
            # Fallback to first available model
            rate_per_second = list(provider_pricing.values())[0]

        return audio_seconds * rate_per_second

    @classmethod
    def calculate_llm_cost(
        cls,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate LLM cost in cents.

        Args:
            provider: LLM provider (openai, anthropic, google)
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in cents
        """
        provider = provider.lower()
        model = model.lower()

        provider_pricing = cls.PRICING["llm"].get(provider, {})
        model_pricing = provider_pricing.get(model, {})

        if not model_pricing and provider_pricing:
            # Try to find a matching model by prefix
            for model_key, pricing in provider_pricing.items():
                if model.startswith(model_key.split("-")[0]):
                    model_pricing = pricing
                    break

        if not model_pricing:
            model_pricing = {"input": 0.0, "output": 0.0}

        # Pricing is per 1K tokens
        input_cost = (input_tokens / 1000) * model_pricing.get("input", 0)
        output_cost = (output_tokens / 1000) * model_pricing.get("output", 0)

        return input_cost + output_cost

    @classmethod
    def calculate_tts_cost(
        cls,
        provider: str,
        model: str,
        characters: int,
    ) -> float:
        """
        Calculate TTS cost in cents.

        Args:
            provider: TTS provider (cartesia, elevenlabs, openai, google)
            model: Model name
            characters: Number of characters

        Returns:
            Cost in cents
        """
        provider = provider.lower()
        model = model.lower()

        provider_pricing = cls.PRICING["tts"].get(provider, {})
        rate_per_1k_chars = provider_pricing.get(model, 0.0)

        if rate_per_1k_chars == 0 and provider_pricing:
            # Fallback to first available model
            rate_per_1k_chars = list(provider_pricing.values())[0]

        return (characters / 1000) * rate_per_1k_chars

    @classmethod
    def calculate_message_cost(
        cls,
        stt_provider: Optional[str] = None,
        stt_model: Optional[str] = None,
        audio_seconds: float = 0,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tts_provider: Optional[str] = None,
        tts_model: Optional[str] = None,
        characters: int = 0,
    ) -> CostBreakdown:
        """
        Calculate total cost for a message.

        Returns:
            CostBreakdown with individual and total costs
        """
        stt_cost = 0.0
        llm_cost = 0.0
        tts_cost = 0.0

        if stt_provider and stt_model and audio_seconds > 0:
            stt_cost = cls.calculate_stt_cost(stt_provider, stt_model, audio_seconds)

        if llm_provider and llm_model and (input_tokens > 0 or output_tokens > 0):
            llm_cost = cls.calculate_llm_cost(
                llm_provider, llm_model, input_tokens, output_tokens
            )

        if tts_provider and tts_model and characters > 0:
            tts_cost = cls.calculate_tts_cost(tts_provider, tts_model, characters)

        return CostBreakdown(
            stt_cost_cents=round(stt_cost, 6),
            llm_cost_cents=round(llm_cost, 6),
            tts_cost_cents=round(tts_cost, 6),
            total_cost_cents=round(stt_cost + llm_cost + tts_cost, 6),
        )

    @classmethod
    def estimate_call_cost(
        cls,
        duration_seconds: float,
        stt_provider: str = "deepgram",
        stt_model: str = "nova-3",
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        tts_provider: str = "cartesia",
        tts_model: str = "sonic-3",
        avg_tokens_per_turn: int = 100,
        avg_chars_per_turn: int = 200,
        turns_per_minute: float = 6,
    ) -> CostBreakdown:
        """
        Estimate cost for a call based on duration.

        This is a rough estimate for pricing UI purposes.
        Actual costs are calculated per-message during the call.

        Args:
            duration_seconds: Call duration in seconds
            turns_per_minute: Average conversation turns per minute

        Returns:
            Estimated CostBreakdown
        """
        duration_minutes = duration_seconds / 60
        total_turns = int(duration_minutes * turns_per_minute)

        # STT: assume 50% of call is user speaking
        stt_seconds = duration_seconds * 0.5
        stt_cost = cls.calculate_stt_cost(stt_provider, stt_model, stt_seconds)

        # LLM: estimate tokens per turn
        total_input_tokens = total_turns * avg_tokens_per_turn
        total_output_tokens = total_turns * avg_tokens_per_turn
        llm_cost = cls.calculate_llm_cost(
            llm_provider, llm_model, total_input_tokens, total_output_tokens
        )

        # TTS: estimate characters per turn
        total_characters = total_turns * avg_chars_per_turn
        tts_cost = cls.calculate_tts_cost(tts_provider, tts_model, total_characters)

        return CostBreakdown(
            stt_cost_cents=round(stt_cost, 4),
            llm_cost_cents=round(llm_cost, 4),
            tts_cost_cents=round(tts_cost, 4),
            total_cost_cents=round(stt_cost + llm_cost + tts_cost, 4),
        )

    @classmethod
    def get_provider_pricing(cls) -> dict:
        """Return the full pricing table for API access."""
        return cls.PRICING
