"""AskMyPDF - Streamlit interface. All RAG logic lives in the rag/ package."""

import html
import itertools
import threading
import time
from datetime import datetime

import streamlit as st

from rag import config
from rag.chain import ask, cited_numbers, cited_sources, clean_citations, is_refusal, summarize
from rag.ingest import (
    clear_database,
    document_stats,
    get_embeddings,
    ingest_pdf,
    rebuild_index,
    remove_document,
)
from rag.llm import (
    OLLAMA,
    LLMError,
    available_providers,
    model_label,
    provider_ready,
    warm_up_ollama,
)

st.set_page_config(page_title="AskMyPDF", page_icon=":material/description:", layout="centered")

SUGGESTIONS = [
    "Summarize the main points of the document",
    "What topics does this document cover?",
    "What are the most important details to know?",
    "List any important numbers, dates or names",
]

CSS = """
<style>
/* Layout */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 820px; padding-top: 2.5rem; }
[data-testid="stSidebarHeader"] { display: none; }
[data-testid="stSidebarUserContent"] { padding-top: 1.75rem; }

/* Sidebar */
.brand { display: flex; align-items: center; gap: .6rem; }
.brand-logo {
  width: 34px; height: 34px; border-radius: 9px; display: grid; place-items: center;
  background: linear-gradient(135deg, #7C83FD, #5A61E0); color: #fff; font-weight: 700; font-size: 15px;
}
.brand-name { font-size: 1.15rem; font-weight: 650; letter-spacing: -.01em; }
.brand-tag { font-size: .8rem; opacity: .6; margin: .35rem 0 0; }
.section-label {
  font-size: .7rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  opacity: .55; margin: 1.4rem 0 .45rem;
}
.status { font-size: .78rem; display: flex; align-items: center; gap: .4rem; margin-top: -.3rem; opacity: .85; }
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.dot.ok { background: #3FB950; box-shadow: 0 0 6px #3FB95088; }
.dot.bad { background: #F85149; }
.doc-name { font-size: .86rem; font-weight: 550; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.doc-meta { font-size: .72rem; opacity: .55; }
.footer { font-size: .72rem; opacity: .45; line-height: 1.6; }

/* Empty state */
.hero { text-align: center; margin: 3.5rem 0 2rem; }
.hero h1 { font-size: 2.1rem; font-weight: 700; letter-spacing: -.02em; margin-bottom: .4rem; }
.hero p { opacity: .65; font-size: 1rem; margin: 0 auto; max-width: 520px; }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; margin-top: 1.5rem; }
.step { border: 1px solid #262D3B; border-radius: .7rem; padding: 1rem; background: #141924; }
.step-num {
  width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center; font-size: .75rem;
  font-weight: 650; background: #7C83FD22; color: #9CA1FF; margin-bottom: .6rem;
}
.step-title { font-weight: 600; font-size: .92rem; }
.step-text { font-size: .8rem; opacity: .6; margin-top: .2rem; }
@media (max-width: 640px) { .steps { grid-template-columns: 1fr; } }

/* Answers */
.chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0 .25rem; }
.chip {
  font-size: .75rem; padding: .2rem .6rem; border-radius: 999px;
  border: 1px solid #7C83FD55; background: #7C83FD14; color: #B4B8FF;
}
.chip b { color: #E6E8EE; font-weight: 600; margin-right: .2rem; }
.thinking { display: flex; align-items: center; gap: .7rem; font-size: .9rem; opacity: .75; padding: .35rem 0; }
.dots { display: inline-flex; gap: 4px; }
.dots i {
  width: 7px; height: 7px; border-radius: 50%; background: #7C83FD;
  animation: pulse 1.2s infinite ease-in-out both;
}
.dots i:nth-child(2) { animation-delay: .15s; }
.dots i:nth-child(3) { animation-delay: .3s; }
@keyframes pulse { 0%, 80%, 100% { opacity: .25; transform: scale(.75); } 40% { opacity: 1; transform: scale(1); } }
.answer-meta { font-size: .72rem; opacity: .45; margin-top: .15rem; }
.refusal {
  display: inline-flex; align-items: center; gap: .5rem; font-size: .92rem; opacity: .8;
  border: 1px dashed #3A4254; border-radius: .6rem; padding: .45rem .8rem;
}
.excerpt-head { font-size: .8rem; font-weight: 600; margin-top: .4rem; }
.excerpt-head span { font-weight: 400; opacity: .55; margin-left: .4rem; }
.excerpt {
  font-size: .82rem; line-height: 1.55; opacity: .85; white-space: pre-wrap;
  border-left: 2px solid #7C83FD66; padding: .2rem .4rem .2rem .8rem; margin: .3rem 0 .9rem;
  max-height: 200px; overflow-y: auto;
}
.note { font-size: .74rem; opacity: .6; margin-top: .3rem; }
.query { font-size: .74rem; opacity: .5; margin-top: .3rem; font-style: italic; }
</style>
"""


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embedding_model():
    return get_embeddings()


@st.cache_data(ttl=10, show_spinner=False)
def cached_provider_status(provider: str) -> tuple[bool, str]:
    return provider_ready(provider)


@st.cache_resource(show_spinner=False)
def start_ollama_warmup() -> threading.Thread:
    """Load the Ollama model in the background once per server, so the first answer is fast."""
    thread = threading.Thread(target=warm_up_ollama, daemon=True)
    thread.start()
    return thread


MODE_LABELS = {
    "similarity": "Similarity (fastest)",
    "mmr": "MMR (more varied excerpts)",
    "rerank": "Rerank (most accurate, slower)",
}


def chat_as_markdown(messages: list[dict]) -> str:
    lines = [f"# AskMyPDF chat · {datetime.now():%Y-%m-%d %H:%M}", ""]
    for m in messages:
        if m["role"] == "user":
            lines += [f"## Q: {m['content']}", ""]
        else:
            lines += [m["content"].strip(), ""]
            if m.get("sources"):
                lines.append("Sources: " + "; ".join(
                    "".join(f"[{n}]" for n in ns) + f" {label}" for ns, label in m["sources"]
                ))
                lines.append("")
    return "\n".join(lines)


def section(label: str) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


def process_uploads(uploads) -> None:
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    progress = st.progress(0.0, text="Processing...")
    for i, upload in enumerate(uploads, start=1):
        progress.progress((i - 1) / len(uploads), text=f"Processing {upload.name}")
        path = config.UPLOAD_DIR / upload.name
        path.write_bytes(upload.getbuffer())
        try:
            n = ingest_pdf(path)
            st.toast(f"{upload.name} processed · {n} chunks", icon=":material/check_circle:")
        except Exception as err:
            st.toast(f"{upload.name}: {err}", icon=":material/error:")
    progress.empty()
    st.session_state.uploader_key += 1  # reset the uploader
    st.rerun()


def delete_document(name: str) -> None:
    remove_document(name)
    (config.UPLOAD_DIR / name).unlink(missing_ok=True)


def render_welcome(has_docs: bool) -> None:
    st.markdown(
        '<div class="hero"><h1>Ask your documents anything</h1>'
        "<p>Upload PDFs and get answers grounded in their content, with the exact file and page "
        "for every claim.</p></div>",
        unsafe_allow_html=True,
    )
    if not has_docs:
        st.markdown(
            '<div class="steps">'
            '<div class="step"><div class="step-num">1</div><div class="step-title">Upload</div>'
            '<div class="step-text">Add one or more text-based PDFs from the sidebar.</div></div>'
            '<div class="step"><div class="step-num">2</div><div class="step-title">Process</div>'
            '<div class="step-text">Documents are split, embedded and indexed locally.</div></div>'
            '<div class="step"><div class="step-num">3</div><div class="step-title">Ask</div>'
            '<div class="step-text">Get cited answers, or a clear "not found".</div></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        return
    st.caption("Try asking")
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, key=f"sugg_{i}", use_container_width=True):
            st.session_state.pending = s
            st.rerun()


def show_thinking(slot, label: str) -> None:
    slot.markdown(
        f'<div class="thinking"><span class="dots"><i></i><i></i><i></i></span>{html.escape(label)}…</div>',
        unsafe_allow_html=True,
    )


def render_text(text: str) -> None:
    if is_refusal(text):
        st.markdown(f'<div class="refusal">ⓘ {html.escape(text.strip())}</div>', unsafe_allow_html=True)
    else:
        st.markdown(text)


def render_sources(msg: dict) -> None:
    sources, chunks = msg.get("sources", []), msg.get("chunks", [])
    if msg.get("query"):
        st.markdown(f'<div class="query">Searched for: {html.escape(msg["query"])}</div>', unsafe_allow_html=True)
    if sources:
        chips = "".join(
            f'<span class="chip"><b>{"".join(f"[{n}]" for n in ns)}</b> {html.escape(label)}</span>'
            for ns, label in sources
        )
        st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
    if msg.get("uncited"):
        st.markdown(
            '<div class="note">The model didn\'t cite specific excerpts, so all retrieved sources are shown.</div>',
            unsafe_allow_html=True,
        )
    meta_col, copy_col = st.columns([6, 1], vertical_alignment="center")
    if msg.get("meta"):
        meta_col.markdown(f'<div class="answer-meta">{html.escape(msg["meta"])}</div>', unsafe_allow_html=True)
    with copy_col.popover("Copy", icon=":material/content_copy:", type="tertiary"):
        st.caption("Use the copy icon in the top-right of the box.")
        st.code(msg["content"].strip(), language=None, wrap_lines=True)
    if chunks:
        with st.expander(f"View source excerpts ({len(chunks)})", icon=":material/format_quote:"):
            for i, c in enumerate(chunks, start=1):
                parts = []
                if "score" in c:  # cosine distance; summaries sample chunks without one
                    parts.append(f"similarity {1 - c['score']:.2f}")
                if "rerank_score" in c:
                    parts.append(f"rerank {c['rerank_score']:.2f}")
                extra = " · ".join(parts)
                st.markdown(
                    f'<div class="excerpt-head">[{i}] {html.escape(c["source"])} · page {c["page"]}'
                    f"<span>{extra}</span></div>"
                    f'<div class="excerpt">{html.escape(c["text"])}</div>',
                    unsafe_allow_html=True,
                )


# ---------- State ----------
load_embedding_model()
st.markdown(CSS, unsafe_allow_html=True)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("top_k", config.TOP_K)
st.session_state.setdefault("mode", config.RETRIEVAL_MODE)
docs = document_stats()


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-logo">A</div><div class="brand-name">AskMyPDF</div></div>'
        '<p class="brand-tag">Answers from your documents, with page citations.</p>',
        unsafe_allow_html=True,
    )

    section("Model")
    provider = st.selectbox("Model", available_providers(), label_visibility="collapsed")
    ok, status = cached_provider_status(provider)
    st.markdown(
        f'<div class="status"><span class="dot {"ok" if ok else "bad"}"></span>'
        f"{html.escape(model_label(provider))} · {status}</div>",
        unsafe_allow_html=True,
    )
    if provider == OLLAMA and ok:
        start_ollama_warmup()

    section("Upload")
    uploads = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploads:
        if st.button(
            f"Process {len(uploads)} file{'s' if len(uploads) > 1 else ''}",
            type="primary",
            icon=":material/bolt:",
            use_container_width=True,
        ):
            process_uploads(uploads)

    section(f"Library · {len(docs)}")
    if not docs:
        st.caption("No documents yet.")
    for name, s in docs.items():
        with st.container(border=True):
            info, summ, remove = st.columns([5, 1, 1], vertical_alignment="center", gap="small")
            info.markdown(
                f'<div class="doc-name" title="{html.escape(name)}">{html.escape(name)}</div>'
                f'<div class="doc-meta">{s["pages"]} pages · {s["chunks"]} chunks</div>',
                unsafe_allow_html=True,
            )
            if summ.button("", icon=":material/summarize:", key=f"sum_{name}", type="tertiary",
                           help="Summarize this document"):
                st.session_state.pending = {"summary": name}
                st.rerun()
            if remove.button("", icon=":material/close:", key=f"del_{name}", type="tertiary", help="Remove"):
                delete_document(name)
                st.rerun()

    scope = None
    if len(docs) > 1:
        choice = st.selectbox("Search in", ["All documents", *docs], key="scope")
        scope = None if choice == "All documents" else choice

    if docs or st.session_state.messages:
        st.write("")
        left, right = st.columns(2)
        if left.button("New chat", icon=":material/add_comment:", use_container_width=True,
                       disabled=not st.session_state.messages):
            st.session_state.messages = []
            st.rerun()
        if right.button("Clear all", icon=":material/delete:", use_container_width=True, disabled=not docs):
            clear_database()
            for f in config.UPLOAD_DIR.glob("*.pdf"):
                f.unlink(missing_ok=True)
            st.session_state.messages = []
            st.rerun()
        if st.session_state.messages:
            st.download_button(
                "Export chat (.md)",
                chat_as_markdown(st.session_state.messages),
                file_name=f"askmypdf-chat-{datetime.now():%Y%m%d-%H%M}.md",
                mime="text/markdown",
                icon=":material/download:",
                use_container_width=True,
            )

    st.divider()
    with st.expander("Settings", icon=":material/tune:"):
        st.slider("Excerpts per answer (top-k)", 2, 10, key="top_k")
        st.selectbox("Retrieval", config.RETRIEVAL_MODES, key="mode", format_func=MODE_LABELS.get)
        if st.session_state.mode == "rerank":
            st.caption("The reranker model (~90 MB) downloads on first use.")
        st.markdown("**Chunking** · applies after rebuilding")
        c1, c2 = st.columns(2)
        new_size = c1.number_input("Chunk size", 200, 3000, config.CHUNK_SIZE, step=100)
        new_overlap = c2.number_input("Overlap", 0, 1000, config.CHUNK_OVERLAP, step=25)
        changed = (new_size, new_overlap) != (config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if st.button("Rebuild index", icon=":material/refresh:", use_container_width=True,
                     disabled=not (changed and docs)):
            if new_overlap >= new_size:
                st.error("Overlap must be smaller than the chunk size.")
            else:
                with st.spinner("Re-chunking documents..."):
                    rebuilt, missing = rebuild_index(int(new_size), int(new_overlap))
                st.toast(f"Rebuilt {len(rebuilt)} document(s) with {new_size}/{new_overlap}",
                         icon=":material/check_circle:")
                if missing:
                    st.toast(f"Removed (original PDF not found, please re-upload): {', '.join(missing)}",
                             icon=":material/warning:")
                st.rerun()

    st.markdown(
        f'<div class="footer">{MODE_LABELS[st.session_state.mode].split(" (")[0]} · top-k {st.session_state.top_k}'
        f" · chunks {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP}<br>"
        f'Embeddings: {config.EMBEDDING_MODEL.split("/")[-1]}</div>',
        unsafe_allow_html=True,
    )


# ---------- Main ----------
question = st.chat_input(
    "Ask a question about your documents" if docs else "Upload a PDF to start asking questions",
    disabled=not docs,
)
pending = st.session_state.pop("pending", None)
summary_of = pending["summary"] if isinstance(pending, dict) else None
question = question or (pending if isinstance(pending, str) else None)
history = [m for m in st.session_state.messages]  # before this turn, for follow-ups
if summary_of:
    st.session_state.messages.append({"role": "user", "content": f"Summarize **{summary_of}**"})
elif question:
    st.session_state.messages.append({"role": "user", "content": question})

# One fixed slot for the welcome screen, so it disappears immediately (instead of
# lingering faded) on the run that starts answering the first question.
welcome = st.empty()
if not st.session_state.messages:
    with welcome.container():
        render_welcome(has_docs=bool(docs))

for msg in st.session_state.messages:
    avatar = ":material/person:" if msg["role"] == "user" else ":material/auto_awesome:"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            render_text(msg["content"])
            render_sources(msg)
        else:
            st.markdown(msg["content"])

if question or summary_of:
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        start = time.perf_counter()
        placeholder = st.empty()
        try:
            if summary_of:
                show_thinking(placeholder, f"Reading {summary_of}")
                answer = summarize(summary_of, provider)
            else:
                follow_up = bool(history) and config.CONDENSE_FOLLOWUPS
                show_thinking(placeholder, "Understanding your follow-up" if follow_up else "Searching your documents")
                answer = ask(
                    question, provider,
                    k=st.session_state.top_k, source=scope, mode=st.session_state.mode,
                    history=[{"role": m["role"], "content": m["content"]} for m in history],
                )
            show_thinking(placeholder, f"Generating answer with {provider.split(' ')[0]}")
            first = next(answer.stream, "")  # waits for the first token
            with placeholder:
                raw = st.write_stream(itertools.chain([first], answer.stream))
            text = raw if is_refusal(raw) else clean_citations(raw, len(answer.sources))
            if text != raw or is_refusal(text):
                with placeholder:
                    render_text(text)  # replace the streamed text with the cleaned version
        except LLMError as err:
            placeholder.error(str(err), icon=":material/error:")
            text = None

        if text:
            meta = f"{model_label(provider)} · {time.perf_counter() - start:.1f}s"
            if scope and not summary_of:
                meta += f" · searched {scope}"
            msg = {"role": "assistant", "content": text, "meta": meta}
            if not summary_of and answer.query.strip().lower() != question.strip().lower():
                msg["query"] = answer.query
            if not is_refusal(text):
                msg["sources"] = cited_sources(text, answer.sources)
                msg["uncited"] = not cited_numbers(text, len(answer.sources))
                msg["chunks"] = [{**d.metadata, "text": d.page_content} for d in answer.sources]
            render_sources(msg)
            st.session_state.messages.append(msg)
