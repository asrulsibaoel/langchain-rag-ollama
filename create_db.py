from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from vectorstore_factory import get_vectorstore

loader = TextLoader("docs/sample.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

vectorstore = get_vectorstore(backend="chroma")
vectorstore.add_documents(docs)
if hasattr(vectorstore, "persist"):
    vectorstore.persist()
