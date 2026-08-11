from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str | None = None
    google_api_key: str | None = None
    admin_api_key: str | None = None
    database_url: str = "sqlite:///./gateway.db"


settings = Settings()
