from pydantic import BaseSettings


class Settings(BaseSettings):
    ENV: str = "dev"

    class Config:
        env_file = ".env"


settings = Settings()