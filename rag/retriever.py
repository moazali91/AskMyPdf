"""Search over the stored chunks: plain similarity, MMR, or cross-encoder reranking."""

from functools import lru_cache

from langchain_core.documents import Document

from rag import config
from rag.ingest import get_vectorstore


@lru_cache(maxsize=1)
def get_reranker():
    """Cross-encoder that scores (question, chunk) pairs; loaded on first use."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANKER_MODEL, device="cpu")


def retrieve(
    question: str,
    k: int | None = None,
    source: str | None = None,
    mode: str | None = None,
) -> list[Document]:
    """Return the k most useful chunks for the question.

    source: only search this PDF.
    mode:   "similarity" (closest vectors), "mmr" (relevant but not repetitive),
            or "rerank" (fetch candidates, re-score them with a cross-encoder).
    Each document's metadata gets "score" = cosine distance (lower = more similar).
    """
    k = k or config.TOP_K
    mode = mode or config.RETRIEVAL_MODE
    where = {"source": source} if source else None
    store = get_vectorstore()

    fetch = k if mode == "similarity" else max(config.FETCH_K, k)
    candidates = store.similarity_search_with_score(question, k=fetch, filter=where)
    for doc, score in candidates:
        doc.metadata["score"] = round(float(score), 3)
    docs = [doc for doc, _ in candidates]

    if mode == "mmr" and docs:
        picked = store.max_marginal_relevance_search(
            question, k=k, fetch_k=fetch, lambda_mult=config.MMR_LAMBDA, filter=where
        )
        by_key = {(d.metadata["source"], d.page_content): d for d in docs}
        return [by_key.get((d.metadata["source"], d.page_content), d) for d in picked]

    if mode == "rerank" and docs:
        scores = get_reranker().predict([(question, d.page_content) for d in docs])
        for d, s in zip(docs, scores):
            d.metadata["rerank_score"] = round(float(s), 3)
        docs = sorted(docs, key=lambda d: d.metadata["rerank_score"], reverse=True)

    return docs[:k]


def format_context(docs: list[Document]) -> str:
    """Number each chunk [1], [2], ... and label it with its file and page for the prompt.

    Small models copy short numeric markers far more reliably than file names,
    so the LLM cites [n] and the app maps n back to the file and page.
    """
    return "\n\n".join(
        f"[{i}] (from {d.metadata['source']}, page {d.metadata['page']})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


if __name__ == "__main__":
    # Usage: python -m rag.retriever [similarity|mmr|rerank] "your question"
    import sys

    args = sys.argv[1:]
    mode = args.pop(0) if args and args[0] in config.RETRIEVAL_MODES else None
    question = " ".join(args) or "What is this document about?"
    for d in retrieve(question, mode=mode):
        m = d.metadata
        print(f"{m['score']:.3f}  {m['source']} p.{m['page']}  {d.page_content[:90]!r}")
