"""Construye la base vectorial (ChromaDB) a partir de los PDFs de la carpeta pdfs/.

Uso:  python build_index.py
Render lo ejecuta durante el build, así el servidor arranca con el índice listo.
"""
import logging
import sys

from rag import RAGConfigError, RAGEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")

if __name__ == "__main__":
    try:
        total = RAGEngine().build_index()
    except RAGConfigError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
    print(f"[OK] Base vectorial creada con {total} fragmentos.")
