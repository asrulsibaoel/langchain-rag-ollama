
# 🧠 LangChain RAG with Ollama, Milvus, Qdrant

This project demonstrates a **Retrieval-Augmented Generation (RAG)** system using **LangChain** and **Ollama**, with pluggable vector databases: **Chroma**, **Milvus**, and **Qdrant**.

---

## 🚀 Features

- 🔍 Document-based Question Answering with LangChain  
- 💬 Local LLM support via Ollama (`llama3`, etc.)  
- 🧠 Vector DB switchable between Chroma / Milvus / Qdrant  
- 🌐 FastAPI backend  
- 🎨 Streamlit web frontend  
- 🐳 Dockerized (with Milvus and Qdrant containers)  

---

## 🧱 Project Structure

```

├── api/                  # FastAPI backend
│   └── main.py
├── docs/
│   └── sample.txt        # Example document
├── vectorstore\_factory.py  # Backend switch logic
├── create\_db.py          # Document ingestion
├── query\_data.py         # CLI QA tool
├── streamlit\_app.py      # Streamlit interface
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md

````

---

## ⚙️ Quick Start

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Docker + Docker Compose

---

### 2. Run via Docker

```bash
docker-compose up --build
````

* FastAPI: [http://localhost:8000/ask?question=What%20is%20LangChain\&backend=qdrant](http://localhost:8000/ask?question=What%20is%20LangChain&backend=qdrant)
* Qdrant UI: [http://localhost:6333](http://localhost:6333)
* Milvus: exposed on port `19530` (gRPC)

---

### 3. Run Manually

```bash
# Create DB (Chroma/Milvus/Qdrant)
python create_db.py

# Ask a question
python query_data.py
```

---

### 4. Use Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

## 🔁 Switch Vector DB

Set the backend as one of:

* `"chroma"`
* `"milvus"`
* `"qdrant"`

You can do this in:

* FastAPI query params
* Streamlit dropdown
* Python code

---

## 🧪 Sample API Call

```http
GET /ask?question=What is LangChain&backend=chroma
```

Response:

```json
{ "answer": "LangChain is a framework..." }
```

---

## 📦 Models Used

* LLM: `llama3` via Ollama
* Embedding: `mxbai-embed-large` via Ollama

Pull the models beforehand:

```bash
ollama pull llama3
ollama pull mxbai-embed-large
```

---

## 🛡 License

MIT — feel free to use and extend.