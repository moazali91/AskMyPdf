# AskMyPDF

Chat with your PDFs and get answers grounded in their content, with the file and page for every claim.

AskMyPDF is a Retrieval-Augmented Generation (RAG) app: it finds the most relevant passages in your documents, then a language model answers using only those passages. If the answer isn't in the documents, it says so instead of guessing.

Runs fully offline with [Ollama](https://ollama.com), or online with [Groq](https://groq.com)'s free API. Total cost: $0.

## Features

- **Page-level citations**: every answer links back to the file and page it came from, with the exact excerpts one click away
- **Refuses when unsure**: replies "I couldn't find that in the uploaded documents" instead of hallucinating
- **Offline or online**: switch between a local Llama 3.2 model (documents never leave your machine) and Groq
- **Reranked retrieval**: a cross-encoder re-scores candidate chunks before answering
- **Follow-up questions**: "what about the second one?" is rewritten into a standalone question before searching
- **Multi-PDF library**: search all documents or one, summarize a document in one click
- **Live settings**: change top-k, retrieval mode and chunking from the sidebar
- **Export**: copy any answer or download the whole chat as Markdown

## How it works

```
Ingestion (once per PDF)
  PDF ─► pypdf (page by page) ─► 1000-char chunks, 150 overlap ─► all-MiniLM-L6-v2 embeddings ─► ChromaDB

Query (every question)
  question ─► rewrite follow-up ─► top-20 similarity search ─► cross-encoder rerank ─► top-4 excerpts
           ─► grounded prompt ─► Ollama / Groq ─► streamed answer with [n] citations ─► file + page
```

Excerpts are numbered `[1]…[4]` in the prompt; the model cites numbers, and the app maps them back to file and page. Citations that point nowhere are removed.

## Evaluation

20-question test set on a 7-page document (15 answerable with known pages, 5 not answerable), chunk size 1000 / overlap 150 / top-k 4. Run with `python -m eval.run_eval`.

| Setup | Hit rate | MRR | Answered | Citation accuracy | Correct refusals | Avg time |
|---|---|---|---|---|---|---|
| Groq, similarity | 93% | 0.76 | 87% | 85% | 100% | 4.9 s |
| **Groq, rerank** | **100%** | **0.93** | **100%** | **100%** | **100%** | 6.0 s |
| Ollama Llama 3.2 3B, similarity | 93% | 0.76 | 73% | 55% | 100% | 44 s |
| Ollama Llama 3.2 3B, rerank | 100% | 0.93 | 100% | 47% | 80% | 54 s |

- **Reranking** gave the largest gain (MRR 0.76 → 0.93). MMR lowered the hit rate (87%), and chunk size mattered little once reranking was on.
- **Llama 3.2 3B** retrieves the right pages but often cites the wrong excerpt number; a larger model fixes this.
- Ollama times were measured on a CPU-only laptop.

Full tables: [eval/results/summary.md](eval/results/summary.md). Per-question results are in the CSVs next to it.

## Getting started

Requires Python 3.11+ (developed on 3.13).

```bash
git clone https://github.com/<your-username>/AskMyPdf.git
cd AskMyPdf
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Pick at least one model provider:

- **Offline (Ollama)**: install [Ollama](https://ollama.com/download), then `ollama pull llama3.2:3b`
- **Online (Groq)**: create a free key at [console.groq.com](https://console.groq.com/keys), copy `.env.example` to `.env` and add the key

Run the app:

```bash
streamlit run app.py
```

The embedding model (~90 MB) and reranker (~90 MB) download on first use.

### Command-line tools

```bash
python -m rag.ingest file.pdf                  # add a PDF to the database
python -m rag.retriever "your question"        # show the retrieved chunks
python -m rag.chain groq "your question"       # answer from the terminal
```

### Running the evaluation

```bash
python -m eval.run_eval                                  # retrieval only
python -m eval.run_eval --provider groq                  # + answers, refusals, timing
python -m eval.run_eval --mode similarity mmr rerank     # compare retrieval modes
python -m eval.run_eval --sweep                          # chunk size / top-k table
python -m eval.run_eval --pdf my.pdf --questions my.csv  # your own document
```

Question files are CSVs with `question,expected_page,answerable`; use `3;5` when an answer appears on several pages.

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub
2. At [share.streamlit.io](https://share.streamlit.io), create an app from the repo with `app.py` as the main file
3. In *Advanced settings*, choose Python 3.13 and add the secrets:
   ```toml
   GROQ_API_KEY = "your_key"
   ON_CLOUD = "1"
   ```

Ollama can't run on Streamlit Cloud, so the hosted app uses Groq only. Cloud storage resets on restart, so visitors upload their own PDFs.

## Project structure

```
app.py               Streamlit interface
rag/
  config.py          all settings (models, chunking, retrieval)
  ingest.py          load PDF → split → embed → store
  retriever.py       similarity, MMR and reranked search
  llm.py             Ollama / Groq clients and error messages
  chain.py           prompts, follow-up rewriting, citations, summaries
eval/
  questions.csv      test set
  run_eval.py        evaluation script
  results/           result tables and per-question CSVs
```

## Tech stack

Python · LangChain · ChromaDB · sentence-transformers (all-MiniLM-L6-v2, ms-marco-MiniLM-L-6-v2 reranker) · Ollama (Llama 3.2 3B) · Groq (gpt-oss-20b) · Streamlit

## Limitations

- Text-based PDFs only; scanned PDFs need OCR (not yet supported)
- English only
- Groq's free tier is rate-limited; the app shows a clear message when the limit is reached
