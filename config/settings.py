from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):

    # ── Environment ──────────────────────────────────────────────────────────
    ENV: str = "dev"

    # ── API Keys (required, loaded from .env) ────────────────────────────────
    OPENAI_API_KEY: str

    # ── Model names ──────────────────────────────────────────────────────────
    SUMMARIZATION_MODEL: str = "gpt-4o-mini"
    QA_MODEL: str = "gpt-4o-mini"
    TOOL_MODEL: str = "gpt-4o-mini"
    WHISPER_MODEL: str = "whisper-1"

    # ── RAG ──────────────────────────────────────────────────────────────────
    CHUNK_MAX_LINES: int = 4        # sentences per chunk
    RAG_TOP_K: int = 3              # chunks retrieved per query
    SUMMARIZATION_TEMPERATURE: float = 0.3

    # ── Semantic cache ───────────────────────────────────────────────────────
    CACHE_SIMILARITY_THRESHOLD: float = 0.85
    CHROMA_PERSIST_DIR: str = "./chroma_cache"

    # ── Pipeline thresholds ──────────────────────────────────────────────────
    ESCALATION_SCORE_THRESHOLD: float = 3.0  # QA overall_score below this → escalate

    # ── Paths ────────────────────────────────────────────────────────────────
    TEMP_AUDIO_PATH: str = "temp_processed.wav"
    PROMPTS_DIR: str = "config/prompts"

    model_config = ConfigDict(
        env_file=".env",
        extra="allow"
    )


settings = Settings()
