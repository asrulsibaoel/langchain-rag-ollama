from uuid import uuid4
from fastapi import FastAPI, HTTPException
from langchain.prompts import PromptTemplate

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


@app.post("/ask")
def ask(data: ChatRequest):
    vs = get_vectorstore(data.backend)
    retriever = vs.as_retriever()
    docs = retriever.get_relevant_documents(data.question)
    context = "\n".join([doc.page_content for doc in docs])

    # Combine history into a text string (optional: format better)
    history_text = "\n".join(
        [f"{turn.role}: {turn.content}" for turn in data.history])

    result = (prompt | llm).invoke({
        "context": context,
        "question": data.question,
        "history": history_text
    })

    return {"answer": result}


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    if len(request.texts) != len(request.metadatas):
        raise HTTPException(
            status_code=400,
            detail="Number of texts and metadatas must match."
        )

    embeddings_model = get_embeddings()
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
    vs = get_vectorstore("milvus")
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
