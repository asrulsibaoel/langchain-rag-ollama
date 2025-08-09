from uuid import uuid4
from fastapi import Depends, FastAPI, HTTPException
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import VectorStore


from src.settings import settings
from src.api.schema import ChatRequest, EmbedResponse, EmbedRequest
from src.llm_factory import get_embeddings, get_llm
from src.vectorstore_factory import get_vectorstore


app = FastAPI()

prompt = PromptTemplate(
    template="""You are an assistant. Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer:""",
    input_variables=["context", "question"]
)
llm = get_llm()

# Dependency: provide a single vectorstore instance


def get_vs() -> VectorStore:
    return get_vectorstore(settings.vector_db)


# Memory store (can be in-memory, Redis, DB, etc.)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create the chain with memory


def get_chain(vs=Depends(get_vs)):
    retriever = vs.as_retriever()
    return ConversationalRetrievalChain.from_llm(
        llm=llm,  # your LLM instance
        retriever=retriever,
        memory=memory
    )


@app.post("/ask")
def ask(req: ChatRequest, chain=Depends(get_chain)):
    result = chain.invoke({"question": req.question})
    return {
        "answer": {
            "content": result["answer"],
            "source_documents": [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata
                } for doc in result.get("source_documents", [])
            ]
        }
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(
    request: EmbedRequest,
    embeddings_model=Depends(get_embeddings),
    vs: VectorStore = Depends(get_vs)
):
    if len(request.texts) != len(request.metadatas):
        raise HTTPException(
            status_code=400,
            detail="Number of texts and metadatas must match."
        )

    vectors = embeddings_model.embed_documents(request.texts)

    # Generate UUIDs
    ids = [str(uuid4()) for _ in request.texts]

    # Ensure metadatas include 'text', 'source', 'category'
    full_metadatas = []
    for i in range(len(request.texts)):
        meta = request.metadatas[i]
        full_metadatas.append({
            "id": ids[i],  # unique identifier
            "text": request.texts[i],  # required text field
            "source": meta.get("source", ""),
            "category": meta.get("category", "")
        })

    # ensure correct field name
    vs.add_texts(
        texts=request.texts,       # needed for LangChain internal processing
        metadatas=full_metadatas,  # must include all non-vector fields
        ids=ids,
        vectors=vectors            # this satisfies the 'vector' field
    )

    return {
        "texts": request.texts,
        "status": "success",
        "inserted": len(request.texts),
        "ids": ids,
        "categories": [meta.get("category", "") for meta in request.metadatas]
    }
