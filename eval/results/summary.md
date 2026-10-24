# AskMyPDF evaluation results

Final runs on eval/questions.csv (20 questions: 15 answerable, 5 not).

## 20261024-164203 · AskMyPDF_Project_Brief.pdf · 20 questions (15 answerable, 5 not) · retrieval only · chunk-size sweep

| Retrieval | Chunk size | Overlap | Top-k | Chunks | Hit rate | MRR |
|---|---|---|---|---|---|---|
| rerank | 500 | 50 | 4 | 36 | 100% | 0.93 |
| rerank | 500 | 50 | 6 | 36 | 100% | 0.93 |
| rerank | 1000 | 150 | 4 | 22 | 100% | 0.93 |
| rerank | 1000 | 150 | 6 | 22 | 100% | 0.93 |
| rerank | 1500 | 200 | 6 | 14 | 100% | 0.90 |

## 20261024-164547 · AskMyPDF_Project_Brief.pdf · 20 questions (15 answerable, 5 not) · retrieval only

| Retrieval | Chunk size | Overlap | Top-k | Chunks | Hit rate | MRR |
|---|---|---|---|---|---|---|
| similarity | 1000 | 150 | 4 | 22 | 93% | 0.76 |
| mmr | 1000 | 150 | 4 | 22 | 87% | 0.72 |
| rerank | 1000 | 150 | 4 | 22 | 100% | 0.93 |

## 20261024-164725 · AskMyPDF_Project_Brief.pdf · 20 questions (15 answerable, 5 not) · Groq (online) (openai/gpt-oss-20b)

| Retrieval | Chunk size | Overlap | Top-k | Chunks | Hit rate | MRR | Answered | Cite acc. | Correct refusals | Avg time |
|---|---|---|---|---|---|---|---|---|---|---|
| similarity | 1000 | 150 | 4 | 22 | 93% | 0.76 | 87% | 85% | 100% | 4.9s |
| rerank | 1000 | 150 | 4 | 22 | 100% | 0.93 | 100% | 100% | 100% | 6.0s |

## 20261024-165226 · AskMyPDF_Project_Brief.pdf · 20 questions (15 answerable, 5 not) · Ollama (offline) (llama3.2:3b)

| Retrieval | Chunk size | Overlap | Top-k | Chunks | Hit rate | MRR | Answered | Cite acc. | Correct refusals | Avg time |
|---|---|---|---|---|---|---|---|---|---|---|
| similarity | 1000 | 150 | 4 | 22 | 93% | 0.76 | 73% | 55% | 100% | 44.3s |
| rerank | 1000 | 150 | 4 | 22 | 100% | 0.93 | 100% | 47% | 80% | 54.4s |

