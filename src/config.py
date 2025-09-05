from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class QdrantConfig(BaseSettings):
    host: str = Field(default="localhost", env="QDRANT_HOST")
    port: int = Field(default=6333, env="QDRANT_PORT")
    api_key: str | None = Field(default=None, env="QDRANT_API_KEY")
    https: bool = Field(default=False, env="QDRANT_HTTPS")

    class Config:
        env_prefix = "QDRANT_"


class EmbeddingConfig(BaseSettings):
    models: List[str] = Field(
        default=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ]
    )
    batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    device: str = Field(default="cuda", env="EMBEDDING_DEVICE")

    class Config:
        env_prefix = "EMBEDDING_"


class DataConfig(BaseSettings):
    corpus_path: str = Field(default="data/beir_data/corpus.jsonl", env="CORPUS_PATH")
    qrels_path: str = Field(default="data/beir_data/qrels.tsv", env="QRELS_PATH")
    query_path: str = Field(default="data/beir_data/query.jsonl", env="QUERY_PATH")
    false_positives_output: str = Field(
        default="data/false_positives.jsonl", env="FALSE_POSITIVES_OUTPUT"
    )

    class Config:
        env_prefix = "DATA_"


class SearchConfig(BaseSettings):
    top_k: int = Field(default=10, env="SEARCH_TOP_K")
    score_threshold: float = Field(default=0.0, env="SEARCH_SCORE_THRESHOLD")

    class Config:
        env_prefix = "SEARCH_"


class AppConfig(BaseSettings):
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)


config = AppConfig()
