import json, os, re
import numpy as np, faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import config
from textclean import clean

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
tok = lambda s: re.findall(r"[a-z0-9\-]+", s.lower())


class Retriever:
    def __init__(self, chunks=None, models=None):
        self.in_memory = chunks is not None
        self.chunks = chunks if chunks is not None else [
            dict(c, text=clean(c["text"])) for c in map(json.loads, open(config.CHUNKS_PATH))]
        self.by_id = {c["id"]: c for c in self.chunks}
        self.embedder, self.reranker = models if models else (
            SentenceTransformer(config.EMBED_MODEL), CrossEncoder(config.RERANK_MODEL))
        self.bm25 = BM25Okapi([tok(c["text"]) for c in self.chunks])
        self._load_index()

    def _load_index(self):
        path = os.path.join(config.INDEX_DIR, "faiss.index")
        if not self.in_memory and os.path.exists(path):   # delete index/ if you re-run ingest.py
            self.index = faiss.read_index(path)
            return
        emb = self.embedder.encode([c["text"] for c in self.chunks], batch_size=32,
                                   normalize_embeddings=True, show_progress_bar=not self.in_memory)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.asarray(emb, dtype="float32"))
        if not self.in_memory:
            os.makedirs(config.INDEX_DIR, exist_ok=True)
            faiss.write_index(self.index, path)

    def dense(self, q, k):
        qv = self.embedder.encode([QUERY_PREFIX + q], normalize_embeddings=True)
        _, idx = self.index.search(np.asarray(qv, dtype="float32"), k)
        return [i for i in idx[0] if i >= 0]

    def sparse(self, q, k):
        return list(np.argsort(-self.bm25.get_scores(tok(q)))[:k])

    def search(self, q, k=5, hybrid=True, pool=30):
        if not hybrid:                       # baseline: dense only, no rerank
            return [dict(self.chunks[i]) for i in self.dense(q, k)]
        fused = {}
        for ranking in (self.dense(q, pool), self.sparse(q, pool)):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0) + 1 / (60 + rank)      # reciprocal rank fusion
        cand = sorted(fused, key=fused.get, reverse=True)[:pool]
        scores = self.reranker.predict([(q, self.chunks[i]["text"]) for i in cand])
        order = np.argsort(-scores)[:k]
        return [dict(self.chunks[cand[j]], score=float(scores[j])) for j in order]
