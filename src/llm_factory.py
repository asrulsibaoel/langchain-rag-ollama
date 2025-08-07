from src.settings import settings


PROVIDER = settings.model_provider.lower()
MODEL_NAME = settings.model_name
EMBEDDING_MODEL = settings.embedding_model
API_KEY = settings.api_key


def get_llm():
    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=MODEL_NAME,
            base_url="http://localhost:11434",
            temperature=0)
    elif PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model_name=MODEL_NAME, temperature=0)
    elif PROVIDER == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(model=settings.model_name, temperature=0, api_key=API_KEY)
    else:
        raise ValueError(f"Unsupported provider: {PROVIDER}")


def get_embeddings():
    if PROVIDER == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        return OllamaEmbeddings(model=EMBEDDING_MODEL)
    # `milvus` is being used as a backend for the
    elif PROVIDER == ["openai", "deepseek"]:
        # retrieval process in the FastAPI
        # application. It is likely being used to
        # store and retrieve vector embeddings for
        # documents or text data to facilitate
        # document retrieval based on user queries.

        from langchain_openai import OpenAIEmbeddings
        kwargs = {"model": EMBEDDING_MODEL}
        if PROVIDER == "deepseek":
            kwargs["base_url"] = "https://api.deepseek.com"
        return OpenAIEmbeddings(**kwargs)
    else:
        raise ValueError(f"Unsupported provider: {PROVIDER}")
