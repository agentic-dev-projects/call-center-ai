from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    ENV: str = "dev"
    OPENAI_API_KEY: str

    model_config = ConfigDict(
        env_file=".env",
        extra="allow"   # allow extra env variables
    )


settings = Settings()