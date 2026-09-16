from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Telecom AI Support Assistant"
    environment: str = "development"

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    # Which backend answers: demo | anthropic | openai.
    # A provider configured without its API key degrades to demo; /health says so.
    llm_provider: str = "demo"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # ------------------------------------------------------------------
    # Routing strategy
    # ------------------------------------------------------------------
    # rules — the application classifies the request and calls one tool.
    #         Deterministic, inspectable, works without a key.
    # agent — the model chooses which tools to call, via real function calling.
    #         Requires a tool-capable provider; falls back to rules otherwise.
    routing_mode: str = "rules"

    # Hard stop on the tool loop, so a model that keeps calling tools cannot
    # spend unbounded time or money on one request.
    agent_max_iterations: int = 5

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    llm_timeout_seconds: float = 30.0
    # Support replies are deliberately short, but thinking tokens count toward
    # this ceiling — leave headroom rather than truncating mid-answer.
    llm_max_tokens: int = 8192
    # Anthropic effort: low | medium | high | xhigh | max. Support routing is
    # not a reasoning-heavy task, and low keeps latency and cost down.
    llm_effort: str = "low"

    # Conversation turns passed back to the model and scanned for identifiers.
    history_max_turns: int = 8

    log_level: str = "INFO"
    # "json" for machine-readable logs, "text" for readable local development.
    log_format: str = "json"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
