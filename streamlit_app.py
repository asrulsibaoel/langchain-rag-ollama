import streamlit as st
from vectorstore_factory import get_vectorstore
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="RAG Chat with Ollama", layout="wide")
st.title("📚 Retrieval-Augmented Generation (RAG) with Ollama")

question = st.text_input("Enter your question:")
backend = st.selectbox("Select Vector Store", ["chroma", "milvus", "qdrant"])

if question:
    with st.spinner("Generating answer..."):
        prompt = PromptTemplate(
            template="""You are an assistant. Use the following context to answer the question.
            Context: {context}
            Question: {question}
            Answer:""",
            input_variables=["context", "question"]
        )
        llm = ChatOllama(model="llama3", temperature=0)
        vs = get_vectorstore(backend)
        retriever = vs.as_retriever()
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])
        response = (prompt | llm).invoke({"context": context, "question": question})
        st.success(response)
