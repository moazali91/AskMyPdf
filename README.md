# AskMyPDF

AskMyPDF is a retrieval-augmented chat app for your own documents. Upload PDFs, ask questions in plain English, and get answers drawn only from those documents, with the file and page number behind every claim. If the answer isn't in the documents, it says so instead of guessing.

It runs fully offline with a local model through Ollama, or online through Groq's free API. Total cost to build and host: $0.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-FF6F61)
![Groq](https://img.shields.io/badge/Groq-gpt--oss--20b-F55036)
![Ollama](https://img.shields.io/badge/Ollama-Llama%203.2%203B-000000?logo=ollama&logoColor=white)

## What The System Does

- Upload one or more text-based PDFs and keep them across restarts
- Answer questions using only retrieved passages, with `[n]` citations mapped to file and page
- Reply "I couldn't find that in the uploaded documents" when the answer is absent
- Show the exact excerpts behind each answer, with similarity and rerank scores
- Rerank candidate passages with a cross-encoder before answering
- Rewrite follow-up questions ("what about the second one?") into standalone questions before searching
- Search all documents or a single PDF, and summarize any document in one click
- Switch between Ollama (offline, documents never leave the machine) and Groq (online, fast)
- Stream answers, copy any answer, and export the whole chat as Markdown
- Change top-k, retrieval mode, chunk size and overlap from the sidebar
- Measure retrieval and answer quality with a built-in evaluation suite

## How It Works

```
Ingestion (once per PDF)
  PDF ─► pypdf, page by page ─► 1000-char chunks, 150 overlap ─► all-MiniLM-L6-v2 ─► ChromaDB on disk

Query (every question)
  question ─► rewrite if follow-up ─► top-20 vector search ─► cross-encoder rerank ─► top-4 excerpts
           ─► grounded prompt ─► Ollama or Groq ─► streamed answer with [n] citations ─► file + page
```

Excerpts are numbered `[1]` to `[4]` in the prompt. The model cites those numbers, the app maps them back to file and page, and removes any citation that points nowhere. Small models copy short numbers far more reliably than file names, which is why citations work this way.

## Evaluation

20 questions on a 7-page document: 15 answerable with known pages, 5 not in the document. Chunk size 1000, overlap 150, top-k 4.

| Setup | Hit rate | MRR | Answered | Citation accuracy | Correct refusals | Avg time |
|---|---|---|---|---|---|---|
| Groq, similarity | 93% | 0.76 | 87% | 85% | 100% | 4.9 s |
| **Groq, rerank** | **100%** | **0.93** | **100%** | **100%** | **100%** | 6.0 s |
| Ollama Llama 3.2 3B, similarity | 93% | 0.76 | 73% | 55% | 100% | 44 s |
| Ollama Llama 3.2 3B, rerank | 100% | 0.93 | 100% | 47% | 80% | 54 s |

- **Hit rate**: answerable questions whose correct page is among the retrieved excerpts
- **MRR**: how high the correct page ranks (1.0 = always first)
- **Citation accuracy**: answered questions that cite an excerpt from the correct page

Findings:

- Reranking gave the largest gain (MRR 0.76 → 0.93) and is the default. MMR lowered the hit rate to 87%, and chunk size mattered little once reranking was on.
- Llama 3.2 3B retrieves the right pages but often cites the wrong excerpt number. A larger model fixes this.
- Ollama times were measured on a CPU-only laptop.

Full tables are in [eval/results/summary.md](eval/results/summary.md).

## Tech Stack

- **Interface**: Streamlit
- **RAG**: LangChain, ChromaDB, pypdf
- **Embeddings**: sentence-transformers `all-MiniLM-L6-v2` (384 dimensions, CPU)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLMs**: Groq `openai/gpt-oss-20b` (online), Ollama `llama3.2:3b` (offline)
- **Config**: python-dotenv

## Repository Layout

```
AskMyPdf/
|-- app.py                 # Streamlit interface
|-- rag/
|   |-- config.py          # all settings: models, chunking, retrieval
|   |-- ingest.py          # load PDF -> split -> embed -> store
|   |-- retriever.py       # similarity, MMR and reranked search
|   |-- llm.py             # Ollama / Groq clients, health checks, error messages
|   `-- chain.py           # prompts, follow-up rewriting, citations, summaries
|-- eval/
|   |-- questions.csv      # test set
|   |-- run_eval.py        # evaluation script
|   `-- results/           # result tables
|-- docs/screenshots/
|-- .streamlit/config.toml # theme and server settings
|-- .env.example
`-- requirements.txt
```

Created at runtime and not committed:

```
data/uploads/              # uploaded PDFs
chroma_db/                 # vector store
.env                       # your API key
```

## Quick Start

### 1. Prerequisites

- Python 3.11+ (developed on 3.13)
- At least one model provider:
  - **Offline**: [Ollama](https://ollama.com/download), then `ollama pull llama3.2:3b`
  - **Online**: a free Groq API key from [console.groq.com/keys](https://console.groq.com/keys)

### 2. Install

```bash
git clone https://github.com/moazali91/AskMyPdf.git
cd AskMyPdf
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

### 3. Configure

Only needed for Groq:

```bash
copy .env.example .env         # Windows
cp .env.example .env           # macOS / Linux
```

Then set `GROQ_API_KEY` in `.env`.

### 4. Run

```bash
streamlit run app.py
```

Open http://localhost:8501, upload a PDF from the sidebar, click **Process**, and ask a question. The embedding model and reranker (about 90 MB each) download on first use.

### 5. Stop

Press `Ctrl+C` in the terminal. Processed documents stay in `chroma_db/` for next time.

## Configuration

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key, required for online mode | empty |
| `OLLAMA_BASE_URL` | Ollama server address | `http://localhost:11434` |
| `ON_CLOUD` | Set to `1` on Streamlit Cloud to hide the Ollama option | `0` |

### Settings in `rag/config.py`

| Setting | Purpose | Default |
|---|---|---|
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Chunking, in characters | `1000`, `150` |
| `TOP_K` | Excerpts given to the model per answer | `4` |
| `RETRIEVAL_MODE` | `similarity`, `mmr` or `rerank` | `rerank` |
| `FETCH_K` | Candidates considered by MMR and the reranker | `20` |
| `RERANKER_MODEL` | Cross-encoder used for reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `OLLAMA_MODEL` | Local model | `llama3.2:3b` |
| `GROQ_MODEL` | Groq model | `openai/gpt-oss-20b` |
| `CONDENSE_FOLLOWUPS` | Rewrite follow-up questions before searching | `True` |

Top-k, retrieval mode and chunking can also be changed live from **Settings** in the sidebar. A new chunk size applies after **Rebuild index**.

## Command-Line Tools

```bash
python -m rag.ingest file.pdf                   # add a PDF to the database
python -m rag.retriever "your question"         # show the retrieved chunks
python -m rag.retriever mmr "your question"     # ...with a specific retrieval mode
python -m rag.chain groq "your question"        # answer from the terminal
```

## Running The Evaluation

```bash
python -m eval.run_eval                                  # retrieval only
python -m eval.run_eval --provider groq                  # + answers, refusals, timing
python -m eval.run_eval --mode similarity mmr rerank     # compare retrieval modes
python -m eval.run_eval --sweep                          # chunk size / top-k table
python -m eval.run_eval --pdf my.pdf --questions my.csv  # your own document
```

Question files are CSVs with the columns `question,expected_page,answerable`. Use `3;5` when an answer appears on several pages. Each run builds its own index in `eval/.chroma_eval/`, so it never touches the app's database, and appends its table to `eval/results/summary.md`.

## Deployment

The app deploys free on [Streamlit Community Cloud](https://share.streamlit.io):

1. Push the repository to GitHub
2. Create an app from the repository with `app.py` as the main file
3. In **Advanced settings**, choose Python 3.13 and add the secrets:
   ```toml
   GROQ_API_KEY = "your_key"
   ON_CLOUD = "1"
   ```

Ollama can't run on Streamlit Cloud, so the hosted app uses Groq only. Cloud storage resets when the app restarts, so visitors upload their own PDFs each session.

## Troubleshooting

**"Ollama not running" in the sidebar**
- Start the Ollama app, or run `ollama serve`
- Check `OLLAMA_BASE_URL` if Ollama runs on another port or machine

**"Model not downloaded"**
- Run `ollama pull llama3.2:3b`, or change `OLLAMA_MODEL` to a model you have

**"No API key" or Groq rejects the key**
- Set `GROQ_API_KEY` in `.env` and restart the app

**Groq rate limit reached**
- The free tier is limited per minute. Wait a minute, or switch to Ollama.

**Groq model unavailable**
- Groq retires models over time. Pick a current one from [console.groq.com/docs/models](https://console.groq.com/docs/models) and update `GROQ_MODEL`.

**Answers are slow on Ollama**
- Ollama on a CPU-only machine takes 20 to 50 seconds per answer
- Lower top-k in Settings, try a smaller model such as `qwen2.5:1.5b`, or use Groq

**A PDF processes but answers say nothing is found**
- The PDF may be scanned images with no text layer. Only text-based PDFs are supported.

**Code changes don't show up**
- Streamlit's file watcher is disabled in `.streamlit/config.toml` because it crashes on the `transformers` package. Restart the app after editing code.

## Known Limitations

- Text-based PDFs only; scanned PDFs need OCR, which isn't supported yet
- English questions and documents only
- Small local models cite less accurately than Groq (see Evaluation)
- The test set is small (20 questions, one 7-page document); a larger set would give more reliable numbers
- Single user, no accounts; everyone using one instance shares the same library
