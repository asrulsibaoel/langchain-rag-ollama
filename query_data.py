from langchain.prompts import PromptTemplate

from src.llm_factory import get_llm
from src.vectorstore_factory import get_vectorstore


prompt = PromptTemplate(
    template="""You are an assistant. Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer:""",
    input_variables=["context", "question"]
)

llm = get_llm()
vectorstore = get_vectorstore(backend="chroma")
retriever = vectorstore.as_retriever()


def ask(question):
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])
    return (prompt | llm).invoke({"context": context, "question": question})


if __name__ == "__main__":
    while True:
        question = input("Question: ")
        print("Answer:", ask(question))
