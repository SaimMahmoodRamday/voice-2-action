from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"

    whisper_model: str = "medium"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # "translate" → Whisper emits English (the extraction model handles English
    # and Roman Urdu well, but not native Urdu script). "transcribe" keeps the
    # source-language script.
    whisper_task: str = "translate"

    upload_dir: str = "./uploads"

    # Optional Notion task-execution tool. Left blank by default → the feature is
    # OFF unless both are set (see app/services/notion.py). Never hardcode these;
    # provide them via the environment / .env.
    notion_token: str = ""
    notion_database_id: str = ""
    # Name of the database's title property (Notion defaults a new DB to "Name").
    notion_title_prop: str = "Name"


settings = Settings()
