"""Grounded answering: retrieve chunks -> build prompt -> stream LLM answer with citations."""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from rag import config
from rag.ingest import get_document_chunks
from rag.llm import complete, friendly_error, get_llm
from rag.retriever import format_context, retrieve

SYSTEM_PROMPT = f"""You are AskMyPDF, an assistant that answers questions using ONLY the numbered context excerpts provided.

Rules:
1. Use only facts stated in the excerpts. Do not use outside knowledge.
2. After each fact, cite the number of the excerpt it came from in square brackets, like [1] or [2][3]. Cite only excerpt numbers, never page numbers or section numbers from the text.
3. If the excerpts do not contain the answer, reply with exactly this sentence and nothing else:
{config.REFUSAL_MESSAGE}
4. Be concise and clear."""

USER_TEMPLATE = """Context excerpts:
{context}

Question: {question}

Answer using only the excerpts above and cite excerpt numbers like [1]. If the answer is not in the excerpts, reply exactly: "{refusal}\""""

SUMMARY_TEMPLATE = """Context excerpts from the document "{source}", in reading order:
{context}

Write a concise summary of this document as 4-8 bullet points covering its main topics. Use only the excerpts above and cite excerpt numbers like [1] after each point."""

CONDENSE_PROMPT = """Rewrite the user's latest question as a standalone question that can be understood without the conversation. Keep the topic the latest question asks about; use the conversation only to fill in what words like "it", "they", "that" or "what about" refer to. If the question is already standalone, return it unchanged. Output only the question, nothing else.

Examples:
Previous question: What is the minimum age for the junior program?
Latest question: And what about the senior one?
Standalone: What is the minimum age for the senior program?

Previous question: Which database does the system use?
Latest question: How is it backed up?
Standalone: How is the system's database backed up?

Previous question: Who wrote the report?
Latest question: What was the sample size?
Standalone: What was the sample size?"""


@dataclass
class Answer:
    """Result of a question. Iterate `stream` to get the text; `sources` are the retrieved chunks.

    `query` is the question actually searched for (differs from the user's
    words when a follow-up was rewritten).
    """

    stream: Iterator[str]
    sources: list[Document] = field(default_factory=list)
    query: str = ""


def build_messages(question: str, docs: list[Document]) -> list:
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=USER_TEMPLATE.format(
                context=format_context(docs),
                question=question,
                refusal=config.REFUSAL_MESSAGE,
            )
        ),
    ]


def _stream_llm(provider: str, messages: list) -> Iterator[str]:
    try:
        llm = get_llm(provider)
        for chunk in llm.stream(messages):
            if chunk.content:
                # gpt-oss sometimes writes citations as 【1】; normalise to [1]
                yield chunk.content.replace("【", "[").replace("】", "]")
    except Exception as err:
        raise friendly_error(err, provider) from err


def condense_question(question: str, history: list[dict], provider: str) -> str:
    """Turn a follow-up ("what about the second one?") into a standalone question.

    history: earlier chat messages as {"role": "user"|"assistant", "content": str}.
    """
    turns = history[-2 * config.HISTORY_TURNS:]
    if not turns:
        return question
    convo = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:600]}" for m in turns
    )
    rewritten = complete(
        provider,
        [
            SystemMessage(content=CONDENSE_PROMPT),
            HumanMessage(content=f"Conversation:\n{convo}\n\nLatest question: {question}"),
        ],
    ).strip().strip('"')
    # Small models sometimes answer instead of rewriting; keep the original then
    if not rewritten or len(rewritten) > 3 * len(question) + 150:
        return question
    return rewritten


def ask(
    question: str,
    provider: str,
    k: int | None = None,
    source: str | None = None,
    mode: str | None = None,
    history: list[dict] | None = None,
) -> Answer:
    """Answer a question from the stored documents.

    source: limit the search to one PDF. history: earlier messages, used to
    rewrite follow-up questions. Raises LLMError (user-friendly message) if the
    LLM fails, either here (rewrite) or while iterating the stream.
    """
    query = question
    if history and config.CONDENSE_FOLLOWUPS:
        query = condense_question(question, history, provider)
    docs = retrieve(query, k=k, source=source, mode=mode)
    if not docs:
        return Answer(stream=iter([config.REFUSAL_MESSAGE]), query=query)
    return Answer(
        stream=_stream_llm(provider, build_messages(query, docs)), sources=docs, query=query
    )


def summarize(source: str, provider: str) -> Answer:
    """Summarize one PDF from excerpts sampled evenly across it."""
    chunks = get_document_chunks(source)
    if len(chunks) > config.SUMMARY_MAX_CHUNKS:
        step = len(chunks) / config.SUMMARY_MAX_CHUNKS
        chunks = [chunks[int(i * step)] for i in range(config.SUMMARY_MAX_CHUNKS)]
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=SUMMARY_TEMPLATE.format(source=source, context=format_context(chunks))),
    ]
    return Answer(stream=_stream_llm(provider, messages), sources=chunks, query=f"Summary of {source}")


def is_refusal(text: str) -> bool:
    return config.REFUSAL_MESSAGE.lower().rstrip(".") in text.lower()


_BRACKET = re.compile(r"\s?\[([^\[\]\n]{1,40})\](?!\()")  # [..] but not markdown links [..](..)


def clean_citations(text: str, n_docs: int) -> str:
    """Keep only real excerpt citations.

    "[1, 3]" and "[Excerpt 2]" become "[1][3]" and "[2]"; citations that point
    nowhere ("[13.1]", "[7]" with 4 excerpts, "[report.pdf, page 3]") are removed.
    Other bracketed text is left alone.
    """

    def fix(m: re.Match) -> str:
        inner = m.group(1).strip()
        nums = re.fullmatch(r"(?:excerpts?\s*)?(\d+(?:\s*[,;&]\s*\d+|\s+and\s+\d+)*)", inner, re.I)
        if nums:
            ns = [int(n) for n in re.findall(r"\d+", nums.group(1))]
            valid = [n for n in ns if 1 <= n <= n_docs]
            return (" " if m.group(0).startswith(" ") else "") + "".join(f"[{n}]" for n in valid) if valid else ""
        if re.fullmatch(r"[\d.\s,;-]+", inner) or re.search(r"page|\.pdf|excerpt|source", inner, re.I):
            return ""
        return m.group(0)

    return _BRACKET.sub(fix, text)


def cited_numbers(text: str, n_docs: int) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= n_docs})


def cited_sources(answer_text: str, docs: list[Document]) -> list[tuple[list[int], str]]:
    """([excerpt numbers], "file, page N") per cited page; excerpts on the same page are merged.

    Falls back to all retrieved excerpts if the model cited none.
    """
    nums = cited_numbers(answer_text, len(docs)) or list(range(1, len(docs) + 1))
    grouped: dict[str, list[int]] = {}
    for n in nums:
        meta = docs[n - 1].metadata
        grouped.setdefault(f"{meta['source']}, page {meta['page']}", []).append(n)
    return [(ns, label) for label, ns in grouped.items()]


if __name__ == "__main__":
    # Usage: python -m rag.chain [ollama|groq] "your question"
    import sys
    import time

    from rag.llm import GROQ, OLLAMA, LLMError

    args = sys.argv[1:]
    provider = OLLAMA
    if args and args[0].lower() in ("ollama", "groq"):
        provider = GROQ if args.pop(0).lower() == "groq" else OLLAMA
    question = " ".join(args) or "What is this document about?"

    start = time.perf_counter()
    answer = ask(question, provider)
    text = ""
    try:
        for token in answer.stream:
            text += token
            print(token, end="", flush=True)
    except LLMError as err:
        print(f"\nERROR: {err}")
        sys.exit(1)
    print(f"\n\n({provider}, {time.perf_counter() - start:.1f}s)")
    if not is_refusal(text):
        text = clean_citations(text, len(answer.sources))
        for ns, label in cited_sources(text, answer.sources):
            print("".join(f"[{n}]" for n in ns), label)
