from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "rag-demo"
    debug: bool = False

    database_url: str = "postgresql://raguser:ragpassword@localhost:5432/ragdemo"

    embedding_model_name: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768
    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 50

    hf_dataset_path: str = ""
    hf_dataset_split: str = "train"
    hf_dataset_text_column: str = "text"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"


settings = Settings()
