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
