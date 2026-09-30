"""Motor RAG de TutorIA.

Equivale al notebook flujo_rag_groq.ipynb, pero empaquetado para una app web:

  PDFs -> chunks -> embeddings -> índice vectorial (numpy) -> búsqueda -> prompt aumentado -> Groq

Para caber en los 512 MB del plan gratuito de Render:
  * Los embeddings se piden a la API de Gemini (gemini-embedding-001): el servidor NO carga
    ningún modelo en memoria.
  * La base vectorial es un arreglo de numpy guardado en disco (en vez de ChromaDB).
    La recuperación sigue siendo vectorial: similitud coseno entre el embedding de la
    pregunta y el de cada fragmento.
"""
import json
import logging
import os
import threading
import time
from pathlib import Path

import httpx
import numpy as np
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from prompts import NO_INFO_MESSAGE, build_messages, format_context

log = logging.getLogger("tutoria.rag")

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = Path(os.getenv("PDF_DIR", BASE_DIR / "pdfs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", BASE_DIR / "index_data"))
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = 768  # gemini-embedding-001 permite reducir de 3072 a 768 sin perder mucho
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
EMBED_BATCH = 50

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


class RAGConfigError(Exception):
    """Falta configuración (por ejemplo una API key o los PDFs)."""


# --------------------------------------------------------------------------- #
# Pasos 1 y 2 — carga de PDFs y chunking
# --------------------------------------------------------------------------- #
def load_pdfs(pdf_dir: Path) -> list[dict]:
    """Paso 1: lee cada página de cada PDF."""
    pages = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"text": text, "source": pdf_path.name, "page": number})
        log.info("%s: %d páginas leídas", pdf_path.name, len(reader.pages))
    return pages


def split_pages(pages: list[dict]) -> list[dict]:
    """Paso 2: divide cada página en fragmentos de ~500 caracteres."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = []
    for page in pages:
        for n, piece in enumerate(splitter.split_text(page["text"])):
            chunks.append(
                {
                    "id": f'{page["source"]}-p{page["page"]}-c{n}',
                    "text": piece,
                    "source": page["source"],
                    "page": page["page"],
                }
            )
    return chunks


# --------------------------------------------------------------------------- #
# Motor
# --------------------------------------------------------------------------- #
class RAGEngine:
    def __init__(self):
        self._init_lock = threading.RLock()  # reentrante: _get_index -> build_index
        self._matrix = None  # (n_fragmentos, dim) normalizada
        self._chunks = None  # lista de dicts con text/source/page
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RAGConfigError(
                    "Falta la variable de entorno GROQ_API_KEY. Créala en console.groq.com."
                )
            self._llm = Groq(api_key=api_key)
        return self._llm

    # ---- Paso 3: embeddings (API de Gemini) ---- #
    def embed(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        """Devuelve una matriz (n, dim) con los vectores normalizados (norma 1).

        task: RETRIEVAL_DOCUMENT para fragmentos del PDF, RETRIEVAL_QUERY para preguntas.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RAGConfigError(
                "Falta la variable de entorno GEMINI_API_KEY. Créala gratis en aistudio.google.com/apikey."
            )
        url = GEMINI_URL.format(model=EMBEDDING_MODEL)
        vectors = []
        for start in range(0, len(texts), EMBED_BATCH):
            part = texts[start : start + EMBED_BATCH]
            body = {
                "requests": [
                    {
                        "model": f"models/{EMBEDDING_MODEL}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task,
                        "outputDimensionality": EMBEDDING_DIM,
                    }
                    for t in part
                ]
            }
            data = self._post_with_retry(url, body, api_key)
            vectors.extend(item["values"] for item in data["embeddings"])

        matrix = np.array(vectors, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)

    @staticmethod
    def _post_with_retry(url: str, body: dict, api_key: str, attempts: int = 4) -> dict:
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        for attempt in range(1, attempts + 1):
            response = httpx.post(url, json=body, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()
            if response.status_code in (429, 500, 503) and attempt < attempts:
                wait = 2**attempt  # 2, 4, 8 s
                log.warning("Gemini respondió %s; reintentando en %ss", response.status_code, wait)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Error de la API de embeddings ({response.status_code}): {response.text[:300]}")
        raise RuntimeError("No se pudo obtener embeddings de Gemini")

    # ---- Paso 4: base vectorial (archivos .npy + .json) ---- #
    def build_index(self) -> int:
        """Lee los PDFs y (re)crea el índice en disco. Devuelve # de fragmentos."""
        if not PDF_DIR.exists() or not list(PDF_DIR.glob("*.pdf")):
            raise RAGConfigError(f"No hay PDFs en '{PDF_DIR}'. Agrega tus documentos ahí.")

        chunks = split_pages(load_pdfs(PDF_DIR))
        if not chunks:
            raise RAGConfigError("Los PDFs no tienen texto extraíble (¿son imágenes escaneadas?).")

        matrix = self.embed([c["text"] for c in chunks], task="RETRIEVAL_DOCUMENT")

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_FILE, matrix)
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        self._matrix = matrix
        self._chunks = chunks
        log.info("Índice creado: %d fragmentos", len(chunks))
        return len(chunks)

    def _get_index(self):
        if self._matrix is None:
            with self._init_lock:
                if self._matrix is None:
                    if EMBEDDINGS_FILE.exists() and CHUNKS_FILE.exists():
                        matrix = np.load(EMBEDDINGS_FILE)
                        if matrix.shape[1] != EMBEDDING_DIM:
                            log.info("El índice tiene otra dimensión: reconstruyéndolo...")
                            self.build_index()
                        else:
                            self._matrix = matrix
                            with open(CHUNKS_FILE, encoding="utf-8") as f:
                                self._chunks = json.load(f)
                            log.info("Índice cargado: %d fragmentos", len(self._chunks))
                    else:
                        log.info("No hay índice: construyéndolo ahora...")
                        self.build_index()  # deja self._matrix y self._chunks listos
        return self._matrix, self._chunks

    def warmup(self):
        """Carga el índice por adelantado (se llama al arrancar)."""
        self._get_index()

    # ---- Paso 5: recuperación (similitud coseno) ---- #
    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        matrix, chunks = self._get_index()
        query_vec = self.embed([query], task="RETRIEVAL_QUERY")[0]
        scores = matrix @ query_vec  # vectores normalizados => producto punto = coseno
        top = np.argsort(scores)[::-1][:k]
        return [
            {
                "text": chunks[i]["text"],
                "source": chunks[i]["source"],
                "page": chunks[i]["page"],
                "score": round(float(scores[i]), 3),
            }
            for i in top
        ]

    # ---- Pasos 6 y 7: prompt aumentado + generación ---- #
    def answer(self, question: str, history: list[dict] | None = None) -> dict:
        history = history or []

        # Si la pregunta es muy corta ("¿y eso?"), le sumamos la pregunta anterior para buscar mejor.
        search_query = question
        if len(question.split()) < 6:
            previous = [t["content"] for t in history if t["role"] == "user"]
            if previous:
                search_query = f"{previous[-1]} {question}"

        fragments = self.retrieve(search_query)
        if not fragments:
            return {"respuesta": NO_INFO_MESSAGE, "fuentes": []}

        messages = build_messages(format_context(fragments), question, history)
        completion = self._get_llm().chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.0,
            max_completion_tokens=2048,
        )
        text = (completion.choices[0].message.content or "").strip() or NO_INFO_MESSAGE
        return {"respuesta": text, "fuentes": fragments}
