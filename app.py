"""Servidor Flask de TutorIA: chat web con arquitectura RAG."""
import logging
import threading

from dotenv import load_dotenv

load_dotenv()  # lee .env en local; en Render las variables ya vienen del entorno

from flask import Flask, jsonify, render_template, request  # noqa: E402

from prompts import TUTOR_MATERIA  # noqa: E402
from rag import GROQ_MODEL, RAGConfigError, RAGEngine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = Flask(__name__)
engine = RAGEngine()

MAX_MESSAGE_CHARS = 1000
MAX_HISTORY_TURNS = 6


def _warmup():
    try:
        engine.warmup()
        app.logger.info("Motor RAG listo")
    except Exception:
        app.logger.exception("No se pudo precalentar el motor RAG")


# Carga el modelo y el índice en segundo plano para que la primera pregunta no espere.
threading.Thread(target=_warmup, daemon=True).start()


def _clean_history(raw) -> list[dict]:
    """Solo acepta turnos válidos (user/assistant con texto) y recorta a los últimos."""
    if not isinstance(raw, list):
        return []
    clean = []
    for turn in raw:
        if (
            isinstance(turn, dict)
            and turn.get("role") in ("user", "assistant")
            and isinstance(turn.get("content"), str)
        ):
            clean.append({"role": turn["role"], "content": turn["content"][:4000]})
    return clean[-MAX_HISTORY_TURNS:]


@app.get("/")
def index():
    return render_template("index.html", materia=TUTOR_MATERIA)


@app.get("/health")
def health():
    return jsonify(status="ok", model=GROQ_MODEL)


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify(error="Escribe una pregunta para empezar."), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify(error=f"La pregunta es muy larga (máximo {MAX_MESSAGE_CHARS} caracteres)."), 400

    try:
        result = engine.answer(message, _clean_history(data.get("history")))
        return jsonify(result)
    except RAGConfigError as exc:
        app.logger.error("Configuración incompleta: %s", exc)
        return jsonify(error=str(exc)), 503
    except Exception:
        app.logger.exception("Error generando la respuesta")
        return jsonify(error="No pude generar la respuesta. Intenta de nuevo en unos segundos."), 502


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
