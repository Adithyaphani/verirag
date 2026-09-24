from sentence_transformers import SentenceTransformer, CrossEncoder
import config
from retriever import Retriever
from graph import build_graph, expand
from verifier import NLIVerifier
from planner import plan
from paperloader import load_paper
import pipeline


class PaperQA:
    def __init__(self):
        self.models = (SentenceTransformer(config.EMBED_MODEL), CrossEncoder(config.RERANK_MODEL))
        self.v = NLIVerifier()
        self.cache = {}

    def _load(self, query):
        meta, chunks = load_paper(query)
        if meta["id"] not in self.cache:
            r = Retriever(chunks=chunks, models=self.models)
            self.cache[meta["id"]] = (meta, r, build_graph(chunks, top_terms=8))
        return self.cache[meta["id"]]

    def ask(self, paper, question, use_verifier=True, max_loops=config.MAX_LOOPS):
        meta, r, G = self._load(paper)
        # reuse the VeriRAG logic by borrowing its methods on a lightweight shell
        shell = pipeline.VeriRAG.__new__(pipeline.VeriRAG)
        shell.r, shell.G, shell.v = r, G, self.v
        res = shell.ask(question, use_verifier=use_verifier, max_loops=max_loops)
        res["paper"] = meta
        return res
