# RAG Document Q&A System

A retrieval-augmented generation (RAG) system that answers questions over a set of research papers. Answers are grounded in the source text, and the system returns the pages it used. Built with LangChain, FAISS, and the Gemini API.

## How it works

1. **Ingest:** Load 3 PDFs (54 pages) and split them into 266 overlapping chunks (1000 characters, 200 overlap).
2. **Embed:** Convert each chunk into a 3072-dimension vector with `gemini-embedding-001`.
3. **Index:** Store the vectors in a FAISS index saved to disk.
4. **Retrieve:** Embed the question and fetch the top-k nearest chunks (k=8).
5. **Generate:** Pass the chunks and the question to `gemini-2.5-flash` with a prompt that restricts answers to the retrieved context and returns "I don't have enough information to answer that." when the answer is missing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with a free API key from aistudio.google.com:

```
GOOGLE_API_KEY=your-key-here
```

## Usage

Put your PDFs in `docs/`, then run:

```bash
python src/ingest.py      # build the FAISS index (run once)
python src/query.py       # ask sample questions
python eval/evaluate.py   # run the 30-question evaluation
```

## Evaluation

Evaluated on a hand-written set of 30 questions (26 answerable, 4
deliberately unanswerable) in `eval/eval_set.json`. Retrieval and generation
are scored separately:

- **Retrieval hit:** the correct source page (±1) appears in the retrieved chunks.
- **Correct:** the expected key phrase appears in the answer (string match).
  For unanswerable questions, the system must refuse.

| Metric | k=4 | k=8 |
|---|---|---|
| Overall accuracy | 27/30 | 30/30 |
| Answerable accuracy | 23/26 | 26/26 |
| Retrieval hit rate | 24/26 | 26/26 |
| Correct refusals | 4/4 | 4/4 |

At k=4, the failures were caused by retrieval: the right chunk did not make the top 4 (two cases), plus one case where the answer sentence was missing from the retrieved chunks. Raising k to 8 fixed all three without breaking any previously correct answer. Typical query latency was about 2-5 seconds.

## Limitations

- The eval set is small and hand-written, and it is graded by string matching, so the results describe behavior on this set, not general accuracy.
- The k=4 run used `gemini-2.5-flash-lite` for questions 1-20 and `gemini-2.5-flash` for questions 21-30 because of free-tier quota limits. The k=8 run used `gemini-2.5-flash` throughout.
- Source PDF quality varies. One paper contains unrelated text, and its noisy extraction hurts retrieval on that document.
- Free-tier rate limits (per minute and per day) slow down evaluation runs.

## Project structure

```
docs/            source PDFs
src/ingest.py    load, chunk, embed, and build the FAISS index
src/query.py     retrieval + generation chain
eval/            evaluation set, script, and saved results
scripts/         one-off check scripts used while building (API key, chunking, retrieval)
```