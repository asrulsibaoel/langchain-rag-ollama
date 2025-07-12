from fastapi import FastAPI, Query
from vectorstore_factory import get_vectorstore
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

app = FastAPI()

prompt = PromptTemplate(
    template="""You are an assistant. Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer:""",
    input_variables=["context", "question"]
)
llm = ChatOllama(model="llama3", temperature=0)

@app.get("/ask")
def ask(question: str = Query(...), backend: str = Query("chroma")):
    vs = get_vectorstore(backend)
    retriever = vs.as_retriever()
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])
    result = (prompt | llm).invoke({"context": context, "question": question})
    return {"answer": result}
