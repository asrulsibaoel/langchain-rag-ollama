import os
from dotenv import load_dotenv

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_provider: str = os.getenv("MODEL_PROVIDER", "ollama")
    model_name: str = os.getenv("MODEL_NAME", "deepseek-r1:8b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "deepseek-r1:8b")
    api_key: str = os.getenv("API_KEY", "")
    vector_db: str = os.getenv("VECTOR_DB", "milvus")

    vector_db_uri: str = os.getenv(
        "VECTOR_DB_URI", "https://in03-ccae31557aae495.serverless.gcp-us-west1.cloud.zilliz.com")
    vector_db_cluster_name: str = os.getenv(
        "VECTOR_DB_CLUSTER_NAME", "cakrul-cv")
    vector_db_collection_name: str = os.getenv(
        "VECTOR_DB_COLLECTION_NAME", "rag_documents")
    vector_db_token: str = os.getenv("VECTOR_DB_TOKEN", "")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()


__all__ = ["settings"]
