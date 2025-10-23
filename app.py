import os, io, json, time, hashlib
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from pathlib import Path

# ---------- Config ----------
load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_DIR = Path(os.getenv("INDEX_DIR", "rag_index"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

INDEX_DIR.mkdir(parents=True, exist_ok=True)
INDEX_BIN = INDEX_DIR / "index.faiss"
META_JSON = INDEX_DIR / "meta.jsonl"
CFG_JSON  = INDEX_DIR / "config.json"

# all-MiniLM-L6-v2 → 384 dims
EMBED_DIM = 384

# ---------- App ----------
app = FastAPI(title="Local RAG Retrieval API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

def require_auth(authorization: Optional[str] = Header(None)):
    if not API_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

# ---------- Embedding model (lazy) ----------
_model = None
def emb_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model

# ---------- Vector index state ----------
index = faiss.IndexFlatIP(EMBED_DIM)  # cosine via normalized vectors
meta: List[Dict[str, Any]] = []

def save_index():
    if index.ntotal == 0:
        return
    faiss.write_index(index, str(INDEX_BIN))
    with open(META_JSON, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    CFG_JSON.write_text(json.dumps({
        "embed_model": EMBED_MODEL_NAME,
        "dim": EMBED_DIM,
        "saved_at": time.time()
    }, indent=2))

def load_index():
    global index, meta
    if INDEX_BIN.exists() and META_JSON.exists():
        index = faiss.read_index(str(INDEX_BIN))
        meta = []
        with open(META_JSON, "r", encoding="utf-8") as f:
            for line in f:
                meta.append(json.loads(line))

load_index()

# ---------- Utils ----------
def sha(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def chunk_text(text: str, target_tokens: int = 800, overlap: int = 120) -> List[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, target_tokens - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+target_tokens])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def _looks_like_pdf(raw: bytes) -> bool:
    return raw[:5] == b"%PDF-"

def extract_text_from_pdf(raw: bytes) -> Tuple[str, str]:
    """
    Return (text, method_used). Robust multi-parser:
      1) pypdf (fast)
      2) PyMuPDF (pymupdf) (very robust)
      3) pdfminer.six (fallback)
    Raises HTTPException if none succeed or file isn't a real PDF.
    """
    if not _looks_like_pdf(raw):
        # Might be HTML or something else
        from fastapi import HTTPException
        preview = raw[:20]
        raise HTTPException(
            status_code=415,
            detail=f"File does not look like a PDF (header={preview!r}). Re-download with curl -L."
        )

    # 1) pypdf (less tolerant, but fast)
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw), strict=False)  # be lenient
        texts = []
        for p in reader.pages:
            t = p.extract_text() or ""
            if t:
                texts.append(t)
        joined = "\n\n".join(texts)
        if joined.strip():
            return joined, "pypdf"
    except Exception:
        pass

    # 2) PyMuPDF (very robust)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=raw, filetype="pdf")
        texts = []
        for page in doc:
            t = page.get_text("text") or ""
            if t:
                texts.append(t)
        joined = "\n\n".join(texts)
        if joined.strip():
            return joined, "pymupdf"
    except Exception:
        pass

    # 3) pdfminer.six (slower but thorough)
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        joined = pdfminer_extract(io.BytesIO(raw)) or ""
        if joined.strip():
            return joined, "pdfminer"
    except Exception:
        pass

    from fastapi import HTTPException
    raise HTTPException(
        status_code=422,
        detail="Unable to extract text from PDF using pypdf, PyMuPDF, or pdfminer. The file may be encrypted or image-only."
    )

def load_text_from_upload(uf: UploadFile, raw: bytes) -> str:
    name = uf.filename.lower()
    if name.endswith(".pdf"):
        text, method = extract_text_from_pdf(raw)
        return text
    elif name.endswith((".txt", ".md")):
        return raw.decode("utf-8", errors="ignore")
    else:
        try:
            return raw.decode("utf-8")
        except:
            raise HTTPException(status_code=415, detail=f"Unsupported file: {uf.filename}")

# ---------- Schemas ----------
class IngestResult(BaseModel):
    added_chunks: int
    files: List[Dict[str, Any]]

class QueryIn(BaseModel):
    query: str
    top_k: int = 5
    min_score: float = 0.0

class RetrievedChunk(BaseModel):
    text: str
    score: float
    meta: Dict[str, Any]

class QueryOut(BaseModel):
    query: str
    results: List[RetrievedChunk]
    embed_model: str

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "embed_model": EMBED_MODEL_NAME,
        "index_size": index.ntotal,
        "meta_count": len(meta),
    }

@app.post("/ingest", response_model=IngestResult, dependencies=[Depends(require_auth)])
async def ingest(files: List[UploadFile] = File(...), source: Optional[str] = Form("uploaded"),
                 chunk_tokens: int = Form(800), overlap: int = Form(120)):
    mdl = emb_model()
    new_vecs = []
    new_meta = []
    file_summ = []

    for uf in files:
        try:
            raw = await uf.read()
            text = load_text_from_upload(uf, raw)
            if not text.strip():
                continue
            doc_id = sha(uf.filename + str(len(raw)))
            chunks = chunk_text(text, target_tokens=int(chunk_tokens), overlap=int(overlap))
            if not chunks:
                continue

            # embed (normalize to use inner-product = cosine)
            embs = mdl.encode(chunks, normalize_embeddings=True)
            if isinstance(embs, list):
                embs = np.array(embs)
            embs = embs.astype("float32")

            new_vecs.append(embs)
            for i, ch in enumerate(chunks):
                new_meta.append({
                    "doc_id": doc_id,
                    "filename": uf.filename,
                    "source": source,
                    "chunk_id": f"{doc_id}-{i}",
                    "text": ch[:4000],  # cap to keep responses small
                    "char_len": len(ch),
                })
            file_summ.append({"filename": uf.filename, "chunks": len(chunks), "status": "ok"})
        except HTTPException as e:
            file_summ.append({"filename": uf.filename, "chunks": 0, "status": f"error: {e.detail}"})
            continue

    if not new_vecs:
        return IngestResult(added_chunks=0, files=file_summ)

    vecs = np.vstack(new_vecs).astype("float32")
    index.add(vecs)
    meta.extend(new_meta)
    save_index()

    return IngestResult(added_chunks=int(vecs.shape[0]), files=file_summ)

@app.post("/query", response_model=QueryOut, dependencies=[Depends(require_auth)])
def query(q: QueryIn):
    print(q.query)
    print(q)
    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="Empty index. Ingest some documents first.")
    mdl = emb_model()
    qv = mdl.encode([q.query], normalize_embeddings=True).astype("float32")
    scores, ids = faiss.Index.search(index, qv, int(q.top_k))  # type: ignore
    results: List[RetrievedChunk] = []
    for s, i in zip(scores[0].tolist(), ids[0].tolist()):
        if i == -1:
            continue
        if s < q.min_score:
            continue
        m = meta[i]
        results.append(RetrievedChunk(
            text=m["text"],
            score=float(s),
            meta={k: v for k, v in m.items() if k != "text"}
        ))
    return QueryOut(query=q.query, results=results, embed_model=EMBED_MODEL_NAME)

@app.post("/reset", dependencies=[Depends(require_auth)])
def reset():
    global index, meta
    index = faiss.IndexFlatIP(EMBED_DIM)
    meta = []
    if INDEX_BIN.exists(): INDEX_BIN.unlink()
    if META_JSON.exists(): META_JSON.unlink()
    if CFG_JSON.exists(): CFG_JSON.unlink()
    return {"ok": True, "message": "Index cleared."}
