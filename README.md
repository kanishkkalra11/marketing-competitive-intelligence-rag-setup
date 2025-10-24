# 🧠 Retrieval-Augmented Generation (RAG) System

A lightweight, local **Retrieval-Augmented Generation (RAG)** backend built with **Python FastAPI**, **Sentence-Transformers**, and **FAISS**.  
This service ingests PDFs or text documents, embeds them into a vector store, and exposes REST endpoints for semantic retrieval — perfect for connecting to LLM agents, n8n, or frontend apps.

---

## 🚀 Features

- **Document ingestion & parsing** for `.pdf`, `.txt`, `.md`
- **Multi-stage extraction** (PyPDF → PyMuPDF → PDFMiner fallback)
- **Semantic embeddings** using `all-MiniLM-L6-v2`
- **Vector store** powered by FAISS (cosine similarity search)
- **REST API** endpoints for ingestion, querying, health checks, and reset
- **Secure** with Bearer token authentication and CORS control
- **Expose easily via ngrok** for external workflows or tools

---

## 🧩 Architecture Overview

```plaintext
Raw Docs (.pdf/.txt)
        ↓
Parsing & Chunking (~800 words, 120 overlap)
        ↓
Embedding (SentenceTransformers MiniLM-L6-v2)
        ↓
FAISS Vector Store (IndexFlatIP)
        ↓
FastAPI Service (/ingest, /query, /health, /reset)
        ↓
ngrok Tunnel → external tools (n8n, Lovable, etc.)
```

---

## 📦 Installation

#### 1. Clone the repository

#### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # on macOS/Linux
# or
.venv\Scripts\activate      # on Windows
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration
Copy .env.example to .env

---

## ▶️ Running the API

Start the FastAPI server:
```bash
uvicorn app:app --reload --port 8000
```

Check that it’s live:
```bash
curl http://127.0.0.1:8000/health
```

---

## 🔄 Ingest Documents

Add your PDFs or text files into a data_raw/ folder, then run:
```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Authorization: Bearer change-this-to-a-long-random-string" \
  -F "files=@data_raw/YourFile.pdf" \
  -F "source=industry_report"
```

---

## 🔍 Query the RAG System

Send a semantic query and retrieve the most relevant document chunks:
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Authorization: Bearer change-this-to-a-long-random-string" \
  -H "Content-Type: application/json" \
  -d '{"query":"latest trends in renewable energy","top_k":5,"min_score":0.2}'
```

Response:
```json
{
  "query": "latest trends in renewable energy",
  "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
  "results": [
    {
      "text": "…chunk of relevant content…",
      "score": 0.83,
      "meta": {
        "filename": "Global_Energy_Report_2025.pdf",
        "source": "industry_report"
      }
    }
  ]
}
```

---

## 🧹 Reset the Index

To clear all embeddings and metadata:
```bash
curl -X POST http://127.0.0.1:8000/reset \
  -H "Authorization: Bearer change-this-to-a-long-random-string"
```

---

🌐 Exposing the API with ngrok

You can use ngrok to securely expose your local FastAPI endpoint to external tools (like n8n or a frontend).

#### 🔗 Setup ngrok
Follow the setup guide here: https://ngrok.com/docs/getting-started

#### 🚀 Start a tunnel
Once ngrok is installed and authed:
```bash
ngrok http 8000
```
Copy the Forwarding URL (e.g. https://xxxx.ngrok-free.app) and use it in your external HTTP requests.

---

## 🔒 Security Notes

- Always protect your /ingest and /reset routes with your Bearer token.
- Restrict allowed origins in .env for production.
- Avoid uploading proprietary or confidential documents unless running locally or on a secure server.

---

## 🧠 Key Components

| Component | Technology | Purpose |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 | Converts text chunks into 384-dim embeddings |
| Vector Store | FAISS (IndexFlatIP) | Efficient similarity search over embeddings |
| API Framework | FastAPI | Lightweight REST service |
| Deployment | ngrok tunnel | External access for tools and frontends |

---

## 📂 Folder Structure

```bash
rag-service/
│
├── app.py               # FastAPI app with ingest/query endpoints
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── rag_index/            # Vector store (FAISS index + metadata)
├── data_raw/             # Raw documents to ingest
└── README.md
```

---

## 🧱 Future Improvements

- Add OCR fallback for scanned PDFs (via pytesseract).
- Implement cross-encoder reranking for better retrieval accuracy.
- Optional cloud deployment (e.g., ChromaDB, Weaviate).
- Build /stats endpoint to monitor index size and file metadata.
