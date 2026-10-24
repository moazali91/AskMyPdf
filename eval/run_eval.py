"""Evaluate AskMyPDF on a fixed question set.

Each experiment builds a fresh, isolated index (eval/.chroma_eval/), so it never
touches the app's chroma_db/ and old chunks never mix in.

Metrics
  hit rate      answerable questions whose expected page is in the top-k chunks
  MRR           mean reciprocal rank of the first chunk from an expected page
  answered      answerable questions that got an answer instead of the refusal
  cite acc.     answered questions whose cited excerpts include an expected page
  refusals      unanswerable questions that got the refusal sentence
  avg time      seconds from question to complete answer

Usage
  python -m eval.run_eval                          # retrieval only, current config
  python -m eval.run_eval --provider groq          # + answers, refusals, timing
  python -m eval.run_eval --sweep                  # tuning table (retrieval only)
  python -m eval.run_eval --sweep --provider groq  # tuning table with answers
  python -m eval.run_eval --mode similarity mmr rerank   # compare retrieval modes
  python -m eval.run_eval --pdf my.pdf --questions my_questions.csv
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from rag import config

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
INDEX_DIR = EVAL_DIR / ".chroma_eval"

# (chunk size, overlap, top-k)
SWEEP = [(500, 50, 4), (500, 50, 6), (1000, 150, 4), (1000, 150, 6), (1500, 200, 6)]

RATE_LIMIT_RETRIES = 6
RATE_LIMIT_WAIT = 20  # seconds


def load_questions(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["answerable"] = r["answerable"].strip().lower() == "yes"
        r["pages"] = {int(p) for p in r["expected_page"].split(";") if p.strip()}
    return rows


def build_index(pdf: Path, chunk_size: int, overlap: int) -> int:
    """Point the rag package at a fresh eval index with these settings and ingest."""
    config.CHUNK_SIZE = chunk_size
    config.CHUNK_OVERLAP = overlap
    config.CHROMA_DIR = INDEX_DIR
    config.COLLECTION_NAME = f"eval_{chunk_size}_{overlap}"

    from rag.ingest import get_vectorstore, ingest_pdf

    get_vectorstore().delete_collection()
    return ingest_pdf(pdf)


def generate(question: str, provider: str, k: int, mode: str):
    """Answer with retries on Groq's free-tier rate limit. Returns (answer, text, seconds)."""
    from rag.chain import ask
    from rag.llm import LLMError

    for attempt in range(RATE_LIMIT_RETRIES):
        start = time.perf_counter()
        try:
            answer = ask(question, provider, k=k, mode=mode)
            text = "".join(answer.stream)
            return answer, text, time.perf_counter() - start
        except LLMError as err:
            if "rate limit" not in str(err).lower() or attempt == RATE_LIMIT_RETRIES - 1:
                raise
            print(f"    rate limited, waiting {RATE_LIMIT_WAIT}s...")
            time.sleep(RATE_LIMIT_WAIT)


def run_experiment(pdf: Path, questions: list[dict], chunk_size: int, overlap: int,
                   k: int, provider: str | None, mode: str) -> tuple[dict, list[dict]]:
    from rag.chain import cited_numbers, clean_citations, is_refusal
    from rag.retriever import retrieve

    n_chunks = build_index(pdf, chunk_size, overlap)
    print(f"\n== {mode}, chunk {chunk_size} / overlap {overlap} / k {k}: {n_chunks} chunks")

    rows = []
    for q in questions:
        docs = retrieve(q["question"], k=k, mode=mode)
        pages = [d.metadata["page"] for d in docs]
        row = {
            "question": q["question"],
            "answerable": q["answerable"],
            "expected_pages": ";".join(map(str, sorted(q["pages"]))),
            "retrieved_pages": ";".join(map(str, pages)),
            "hit": None,
            "rank": None,
        }
        if q["answerable"]:
            rank = next((i for i, p in enumerate(pages, 1) if p in q["pages"]), None)
            row["hit"], row["rank"] = rank is not None, rank

        if provider:
            answer, text, seconds = generate(q["question"], provider, k, mode)
            text = clean_citations(text, len(answer.sources))
            refused = is_refusal(text)
            # Only explicit citations count; an uncited answer gets no credit
            cited = {answer.sources[n - 1].metadata["page"] for n in cited_numbers(text, len(answer.sources))}
            row.update(
                answer=text.strip().replace("\n", " "),
                refused=refused,
                cited_pages=";".join(map(str, sorted(cited))) if not refused else "",
                cite_ok=(not refused and bool(cited & q["pages"])) if q["answerable"] else None,
                seconds=round(seconds, 2),
            )
            mark = "refused" if refused else "answered"
            print(f"  [{mark:8}] {seconds:5.1f}s  {q['question']}")
        rows.append(row)

    return summarize(rows, chunk_size, overlap, k, n_chunks, provider, mode), rows


def summarize(rows, chunk_size, overlap, k, n_chunks, provider, mode) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    s = {
        "mode": mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "k": k,
        "chunks": n_chunks,
        "hit_rate": sum(r["hit"] for r in ans) / len(ans),
        "mrr": sum(1 / r["rank"] for r in ans if r["rank"]) / len(ans),
        "provider": provider or "-",
    }
    if provider:
        answered = [r for r in ans if not r["refused"]]
        s["answered"] = len(answered) / len(ans)
        s["cite_acc"] = (sum(r["cite_ok"] for r in answered) / len(answered)) if answered else 0.0
        s["refusals"] = sum(r["refused"] for r in unans) / len(unans) if unans else 0.0
        s["avg_time"] = sum(r["seconds"] for r in rows) / len(rows)
    return s


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def markdown_table(summaries: list[dict]) -> str:
    gen = "answered" in summaries[0]
    head = ["Retrieval", "Chunk size", "Overlap", "Top-k", "Chunks", "Hit rate", "MRR"]
    if gen:
        head += ["Answered", "Cite acc.", "Correct refusals", "Avg time"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in summaries:
        cells = [s["mode"], s["chunk_size"], s["overlap"], s["k"], s["chunks"], pct(s["hit_rate"]), f"{s['mrr']:.2f}"]
        if gen:
            cells += [pct(s["answered"]), pct(s["cite_acc"]), pct(s["refusals"]), f"{s['avg_time']:.1f}s"]
        lines.append("| " + " | ".join(map(str, cells)) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AskMyPDF retrieval and answers.")
    parser.add_argument("--pdf", default=str(config.BASE_DIR / "AskMyPDF_Project_Brief.pdf"))
    parser.add_argument("--questions", default=str(EVAL_DIR / "questions.csv"))
    parser.add_argument("--provider", choices=["groq", "ollama"], help="also generate answers")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--sweep", action="store_true", help="run the tuning table")
    parser.add_argument("--mode", nargs="+", choices=config.RETRIEVAL_MODES,
                        default=[config.RETRIEVAL_MODE], help="retrieval mode(s) to compare")
    args = parser.parse_args()

    from rag.llm import GROQ, OLLAMA

    provider = {"groq": GROQ, "ollama": OLLAMA}.get(args.provider)
    pdf, questions = Path(args.pdf), load_questions(Path(args.questions))
    configs = SWEEP if args.sweep else [(args.chunk_size, args.overlap, args.k)]

    summaries = []
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)
    for (chunk_size, overlap, k), mode in [(c, m) for c in configs for m in args.mode]:
        summary, rows = run_experiment(pdf, questions, chunk_size, overlap, k, provider, mode)
        summaries.append(summary)
        out = RESULTS_DIR / f"{stamp}_{mode}_c{chunk_size}_o{overlap}_k{k}_{args.provider or 'retrieval'}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    table = markdown_table(summaries)
    n_ans = sum(q["answerable"] for q in questions)
    header = (
        f"## {stamp} · {pdf.name} · {len(questions)} questions "
        f"({n_ans} answerable, {len(questions) - n_ans} not) · "
        f"{provider or 'retrieval only'}"
        + (f" ({config.GROQ_MODEL if provider == GROQ else config.OLLAMA_MODEL})" if provider else "")
    )
    print(f"\n{header}\n\n{table}\n")
    with open(RESULTS_DIR / "summary.md", "a", encoding="utf-8") as f:
        f.write(f"{header}\n\n{table}\n\n")
    print(f"Per-question results saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
