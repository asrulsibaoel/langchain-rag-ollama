from langchain.vectorstores import Chroma, Qdrant, Milvus
from langchain_ollama.embeddings import OllamaEmbeddings
from typing import Literal

def get_vectorstore(backend: Literal["chroma", "milvus", "qdrant"], persist_directory="db"):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    if backend == "chroma":
        return Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    elif backend == "milvus":
        return Milvus(
            embedding_function=embeddings,
            connection_args={"host": "localhost", "port": "19530"},
            collection_name="rag_documents"
        )
    elif backend == "qdrant":
        return Qdrant(
            embedding_function=embeddings,
            url="http://localhost:6333",
            collection_name="rag_documents"
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
