import os

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"   # used inside the agent
EVAL_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"          # different model, used only for scoring

CHUNKS_PATH = "chunks.jsonl"
INDEX_DIR = "index"
HF_TOKEN = "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"  # free token: huggingface.co/settings/tokens

ENTAIL_THRESHOLD = 0.5
MAX_LOOPS = 2