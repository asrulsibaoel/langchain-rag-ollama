from typing import Literal
from langchain_community.vectorstores import Chroma, Qdrant, Milvus

# from langchain_ollama import OllamaEmbeddings

from src.llm_factory import get_embeddings

from src.settings import settings


def get_vectorstore(backend: Literal["chroma", "milvus", "qdrant"], persist_directory="db"):
    embeddings = get_embeddings()

    if backend == "chroma":
        return Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    elif backend == "milvus":
        return Milvus(
            embedding_function=embeddings,
            connection_args={
                "uri": settings.vector_db_uri,
                "token": settings.vector_db_token,
                "db_name": settings.vector_db_cluster_name,
            },
            collection_name=settings.vector_db_collection_name,
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {
                    "nlist": 128
                }
            },
            consistency_level="Strong",
            drop_old=False,
            vector_field="vector",
            text_field="text",
        )
    elif backend == "qdrant":
        return Qdrant(
            embedding_function=embeddings,
            url="http://localhost:6333",
            collection_name="rag_documents"
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
