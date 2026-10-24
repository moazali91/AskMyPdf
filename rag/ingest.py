"""Ingestion pipeline: load PDF -> split into chunks -> embed -> store in Chroma."""

import json
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once and reuse it."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore() -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_pdf(pdf_path: Path) -> list[Document]:
    """Read a PDF page by page, keeping the file name and a 1-based page number."""
    reader = PdfReader(str(pdf_path))
    docs = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue  # skip empty / image-only pages
        docs.append(
            Document(
                page_content=text,
                metadata={"source": pdf_path.name, "page": page_num},
            )
        )
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_pdf(pdf_path: str | Path) -> int:
    """Process one PDF into the vector store. Returns the number of chunks stored.

    Chunk IDs are "<file name>-<index>", so re-ingesting the same file
    overwrites its chunks instead of duplicating them.
    """
    pdf_path = Path(pdf_path)
    pages = load_pdf(pdf_path)
    if not pages:
        raise ValueError(
            f"No text found in {pdf_path.name}. It may be a scanned/image-only PDF."
        )

    chunks = split_documents(pages)
    ids = [f"{pdf_path.name}-{i}" for i in range(len(chunks))]

    store = get_vectorstore()
    remove_document(pdf_path.name, store)  # drop stale chunks if the file changed
    store.add_documents(chunks, ids=ids)
    return len(chunks)


def remove_document(source: str, store: Chroma | None = None) -> None:
    store = store or get_vectorstore()
    existing = store.get(where={"source": source})
    if existing["ids"]:
        store.delete(ids=existing["ids"])


def list_documents() -> list[str]:
    """Names of all PDFs currently stored."""
    metas = get_vectorstore().get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metas if m and "source" in m})


def document_stats() -> dict[str, dict]:
    """{file name: {"pages": highest page with text, "chunks": chunk count}}."""
    stats: dict[str, dict] = {}
    for m in get_vectorstore().get(include=["metadatas"])["metadatas"]:
        if not m or "source" not in m:
            continue
        s = stats.setdefault(m["source"], {"pages": 0, "chunks": 0})
        s["chunks"] += 1
        s["pages"] = max(s["pages"], int(m.get("page", 0)))
    return dict(sorted(stats.items()))


def get_document_chunks(source: str) -> list[Document]:
    """All chunks of one PDF, in reading order."""
    data = get_vectorstore().get(where={"source": source}, include=["documents", "metadatas"])
    order = sorted(range(len(data["ids"])), key=lambda i: int(data["ids"][i].rsplit("-", 1)[1]))
    return [Document(page_content=data["documents"][i], metadata=data["metadatas"][i]) for i in order]


def rebuild_index(chunk_size: int, chunk_overlap: int) -> tuple[list[str], list[str]]:
    """Re-chunk every uploaded PDF with new settings. Returns (rebuilt, missing).

    "missing" are stored documents whose PDF is no longer in data/uploads
    (e.g. added from the command line); they are dropped from the index.
    """
    names = list(document_stats())
    config.CHUNK_SIZE, config.CHUNK_OVERLAP = chunk_size, chunk_overlap
    clear_database()
    rebuilt, missing = [], []
    for name in names:
        path = config.UPLOAD_DIR / name
        if path.exists():
            ingest_pdf(path)
            rebuilt.append(name)
        else:
            missing.append(name)
    config.INDEX_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.INDEX_SETTINGS_FILE.write_text(
        json.dumps({"chunk_size": chunk_size, "chunk_overlap": chunk_overlap})
    )
    return rebuilt, missing


def clear_database() -> None:
    """Remove every stored chunk."""
    store = get_vectorstore()
    store.delete_collection()


if __name__ == "__main__":
    # Usage: python -m rag.ingest path/to/file.pdf [more.pdf ...]
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m rag.ingest <file.pdf> [...]")
        sys.exit(1)
    for arg in sys.argv[1:]:
        n = ingest_pdf(arg)
        print(f"{Path(arg).name}: {n} chunks stored")
    print("Documents in database:", list_documents())
