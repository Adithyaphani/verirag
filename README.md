# VeriRAG: a self-auditing research agent

Ask a question about an arXiv paper and get an answer where **every sentence is checked against the source text**. Unsupported sentences trigger re-retrieval and a rewrite. If the evidence still isn't there, the sentence is dropped, or the system says "Insufficient evidence" instead of guessing.

Built only with free tools: open Hugging Face models, FAISS, NetworkX and Gradio.

## The problem

A normal RAG system retrieves text and lets an LLM write the answer, with no check that the answer is backed by the sources. VeriRAG adds a verification step, so each claim can be traced to a specific passage.

## How it works

```
question (+ paper name / arXiv ID)
   |
1. PLANNER      splits a hard question into 1-3 sub-questions
2. RETRIEVER    dense (bge-small) + BM25 -> RRF fusion -> cross-encoder rerank
                concept graph adds chunks that share key terms
3. GENERATOR    Llama-3.1-8B writes an answer citing chunk IDs, e.g. [1706.03762v7_9]
4. VERIFIER     splits the answer into claims; a DeBERTa NLI model checks each claim
                against its cited chunk
                  entailed        -> keep
                  not entailed    -> retrieve again with the claim, regenerate (max 2 loops)
5. OUTPUT       verified sentences + confidence line + a per-claim audit table
```

## Models

| Role | Model | Runs where |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` | local CPU |
| Reranker | `BAAI/bge-reranker-base` | local CPU |
| Claim verifier | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | local CPU |
| Eval judge | `cross-encoder/nli-deberta-v3-base` | local CPU |
| Planner + answer writer | `meta-llama/Llama-3.1-8B-Instruct` | HF Inference Providers (needs `HF_TOKEN`) |

Only the LLM needs the token. Everything else is downloaded once and runs locally.

## Two ways to use it

1. **Any arXiv paper** (`app_paper.py`): enter a title, arXiv ID or URL plus a question. The paper is fetched, chunked and indexed on the fly, then cached in memory.
2. **Fixed corpus** (`app.py`): about 30 quantum error correction papers ingested ahead of time.

## Project structure

```
verirag/
├── requirements.txt
├── config.py         # model names, entailment threshold, paths
├── ingest.py         # arXiv search -> PDFs -> chunks.jsonl (fixed corpus)
├── textclean.py      # fixes PDF ligatures, hyphenation, [12]-style refs
├── retriever.py      # FAISS + BM25 + RRF + reranker (disk or in-memory)
├── graph.py          # concept graph (chunks <-> TF-IDF key terms)
├── llm.py            # HF Inference client
├── planner.py        # question -> sub-questions
├── verifier.py       # claim splitting + NLI entailment
├── pipeline.py       # VeriRAG: plan > retrieve > answer > verify > correct
├── paperloader.py    # arXiv lookup + PDF download + chunking (any paper)
├── paperqa.py        # PaperQA: VeriRAG over a single paper, with caching
├── evaluate.py       # baseline vs hybrid vs full, scored by an independent NLI model
├── app_paper.py      # Gradio UI: paper + question
├── app.py            # Gradio UI: fixed corpus
├── data/eval_questions.json
├── papers/           # downloaded PDFs (created automatically)
├── chunks.jsonl      # created by ingest.py
└── index/            # FAISS index for the fixed corpus (created on first run)
```

## Requirements

- Python 3.11 or 3.12 recommended (3.14 works but prints a harmless `torch.jit` warning)
- About 2 GB free disk for models
- A free Hugging Face account with a **fine-grained token that has the "Make calls to Inference Providers" permission**. A plain Read token returns 403.

See `COMMANDS.md` for the exact commands.

## Reading the output

The UI shows one confidence line, for example:

`100% grounded | first draft 50% | avg entailment 0.87 | 2 loop(s)`

- **Entailment**: the NLI model's probability (0 to 1) that the cited chunk supports a claim. At or above 0.5 (`ENTAIL_THRESHOLD`) the claim counts as supported.
- **Grounded**: the share of claims in the final answer that are supported. It is often 100% because unsupported sentences are rewritten or dropped before the final check.
- **First draft**: the share of claims that were supported in the LLM's first attempt, before any correction. This is the number that shows the verifier loop doing work.
- **Avg entailment**: the mean entailment score over the final claims. It varies continuously, so it is better for comparing answers.
- **Loops**: how many re-retrieve-and-regenerate rounds were needed (max 2).
- With the **Verifier loop** box unticked, the line reads `Verifier off` and the answer is unaudited.

## Limitations

- **Grounded does not mean correct.** The verifier checks that the cited text supports a sentence, not that the sentence answers the question or that the paper is right.
- **PDF extraction is imperfect.** Equations and tables are flattened and can come out garbled. A claim that matches garbled text still passes.
- **Chunk-level checking.** A claim that combines facts from two chunks can score low against either one.
- **Title search takes the top arXiv hit.** Vague titles can match the wrong paper. An arXiv ID is always exact.
- **The free HF tier has limited credits.** `evaluate.py` makes many LLM calls, so start with a small question set.
- **Evaluation is faithfulness only.** It measures support by the retrieved text, not answer correctness. Report it that way.

## Results

Not yet measured. After running `evaluate.py`, add the real numbers from `results.json` here.

## Ideas for extension

- Per-claim relevance check (does the sentence answer the question?)
- HTML (ar5iv) fallback for cleaner text than PDF
- Multi-paper questions and contradiction finding
- A local LLM fallback when free credits run out
- Deploy the UI on a free Hugging Face Space

## License

MIT (placeholder, change as you prefer)
