"""Central settings for AskMyPDF. Change values here, not in the code."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Hugging Face cache works without symlinks on Windows; hide the noisy warning
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "askmypdf"
# Chunk settings the current index was built with (written when rebuilt from the app)
INDEX_SETTINGS_FILE = CHROMA_DIR / "index_settings.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 4
RETRIEVAL_MODES = ["similarity", "mmr", "rerank"]
RETRIEVAL_MODE = "rerank"  # best in eval/results/summary.md
FETCH_K = 20  # candidates considered by MMR and the reranker
MMR_LAMBDA = 0.5  # 1 = pure relevance, 0 = pure diversity
# "BAAI/bge-reranker-base" is stronger but ~1.1 GB, too heavy for Streamlit Cloud
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CONDENSE_FOLLOWUPS = True
HISTORY_TURNS = 3  # previous question/answer pairs used to rewrite a follow-up

SUMMARY_MAX_CHUNKS = 8

OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_KEEP_ALIVE = "30m"
# Groq retires models over time; see console.groq.com/docs/models
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_REASONING_EFFORT = "low"  # gpt-oss is a reasoning model; low keeps it fast
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TEMPERATURE = 0.0

# On Streamlit Community Cloud Ollama is unavailable, so only Groq is offered
ON_CLOUD = os.getenv("ON_CLOUD", "0") == "1"

REFUSAL_MESSAGE = "I couldn't find that in the uploaded documents."


def load_index_settings() -> None:
    """Use the chunk settings the stored index was built with, if it was rebuilt from the app."""
    global CHUNK_SIZE, CHUNK_OVERLAP
    try:
        saved = json.loads(INDEX_SETTINGS_FILE.read_text())
        CHUNK_SIZE, CHUNK_OVERLAP = int(saved["chunk_size"]), int(saved["chunk_overlap"])
    except (OSError, ValueError, KeyError):
        pass


load_index_settings()
