from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Telecom AI Support Assistant"
    environment: str = "development"
    llm_provider: str = "demo"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""

    # Seconds to wait on the LLM provider before falling back to demo mode.
    llm_timeout_seconds: float = 20.0

    # Conversation turns passed back into the model and scanned for identifiers.
    history_max_turns: int = 8

    log_level: str = "INFO"
    # "json" for machine-readable logs, "text" for readable local development.
    log_format: str = "json"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
