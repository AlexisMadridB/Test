# TutorIA — Tutor académico con RAG

Chatbot web (Flask) que responde preguntas usando **únicamente** los documentos PDF del curso.
Cada respuesta cita el documento y la página de donde salió la información.

## Arquitectura RAG

```
PDFs (carpeta pdfs/)
   │  pypdf
   ▼
Chunking (RecursiveCharacterTextSplitter, 500 caracteres, solape 50)
   │
   ▼
Embeddings (paraphrase-multilingual-MiniLM-L12-v2, 384 dimensiones, local)
   │
   ▼
ChromaDB (similitud coseno)  ◄── Pregunta del usuario (también convertida en embedding)
   │  top-k fragmentos
   ▼
Prompt aumentado (system prompt + <contexto> + <pregunta>, delimitados con XML)
   │
   ▼
LLM en Groq (openai/gpt-oss-120b, temperature 0)
   │
   ▼
Respuesta con citas  →  interfaz web Flask
```

### Del notebook al código

| Paso del notebook (`flujo_rag_groq.ipynb`) | Dónde está |
|---|---|
| 1. Carga de PDFs | `rag.py` → `load_pdfs()` |
| 2. Chunking | `rag.py` → `split_pages()` |
| 3. Embeddings | `rag.py` → `RAGEngine.embed()` |
| 4. Base vectorial ChromaDB | `rag.py` → `RAGEngine.build_index()` |
| 5. Recuperación | `rag.py` → `RAGEngine.retrieve()` |
| 6. Prompt aumentado | `prompts.py` → `build_messages()` |
| 7. Generación con Groq | `rag.py` → `RAGEngine.answer()` |
| Pipeline completo (`rag_pipeline`) | `RAGEngine.answer()` |

**Diferencia con el notebook:** los embeddings usan `fastembed` (mismo modelo, pero con ONNX en vez de PyTorch).
Es más liviano y permite correr en los 512 MB del plan gratuito de Render. La evaluación con RAGAs del paso 8
no se incluyó porque es una prueba de laboratorio y no parte de la app.

## Estructura

```
├── app.py            # Servidor Flask (rutas /, /api/chat, /health)
├── rag.py            # Motor RAG
├── prompts.py        # System prompt, few-shot y delimitadores
├── build_index.py    # Crea la base vectorial desde pdfs/
├── pdfs/             # Documentos base del tutor  ← pon tus PDFs aquí
├── templates/        # index.html
├── static/           # style.css, chat.js
├── requirements.txt
├── render.yaml       # Configuración de despliegue en Render
└── .env.example
```

## Ejecutar en local

```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # En Windows: copy .env.example .env
# edita .env y pega tu GROQ_API_KEY (gratis en https://console.groq.com)

python build_index.py           # indexa los PDFs (la 1.ª vez descarga el modelo, ~220 MB)
python app.py                   # abre http://127.0.0.1:5000
```

## Configuración

| Variable | Para qué sirve | Valor por defecto |
|---|---|---|
| `GROQ_API_KEY` | Key de Groq (obligatoria) | — |
| `GROQ_MODEL` | Modelo de Groq | `openai/gpt-oss-120b` |
| `TUTOR_MATERIA` | Materia que aparece en la interfaz y en el prompt | `Bases de Datos` |
| `RAG_TOP_K` | Fragmentos recuperados por pregunta | `5` |

## Desplegar en Render

1. Sube el proyecto a GitHub (con la carpeta `pdfs/` incluida). **No subas el `.env`**; ya está en `.gitignore`.
2. En [render.com](https://render.com): **New → Blueprint** y elige el repositorio (usa `render.yaml`).
   Alternativa manual: **New → Web Service** con estos valores:
   - Build Command: `pip install -r requirements.txt && python build_index.py`
   - Start Command: `gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`
   - Instance type: Free
3. En **Environment** agrega `GROQ_API_KEY` con tu key y `PYTHON_VERSION` = `3.11.9`.
4. Espera el deploy. La URL pública queda como `https://<tu-servicio>.onrender.com`.

> El plan gratuito de Render duerme el servicio tras ~15 minutos sin uso. La primera visita después de eso
> tarda entre 30 y 60 segundos en despertar (carga el modelo de embeddings otra vez).

## Notas

- Si cambias los PDFs, vuelve a correr `python build_index.py` (en Render, un nuevo deploy lo hace solo).
- Si el tutor responde "No encontré información…", la respuesta no está en los PDFs: es el comportamiento esperado.
