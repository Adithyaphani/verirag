import config
from retriever import Retriever
from graph import build_graph, expand
from verifier import NLIVerifier
from planner import plan
from llm import chat

GEN_SYS = ("Answer using ONLY the evidence chunks provided. After every factual sentence, cite the "
           "supporting chunk id in square brackets BEFORE the period, like: Surface codes need many "
           "qubits [2101.01234v1_7]. If the evidence does not answer the question, reply exactly: "
           "The evidence does not contain enough information. Never use outside knowledge. "
           "Write 3-6 sentences.")
ABSTAIN = "Insufficient evidence to answer reliably."


class VeriRAG:
    def __init__(self):
        self.r = Retriever()
        self.G = build_graph(self.r.chunks)
        self.v = NLIVerifier()

    def _gather(self, queries, hybrid, use_graph, per_q=4):
        ev = {}
        for q in queries:
            for c in self.r.search(q, k=per_q, hybrid=hybrid):
                ev.setdefault(c["id"], c)
        if use_graph:
            for cid in expand(self.G, list(ev), set(ev), n=2):
                ev.setdefault(cid, self.r.by_id[cid])
        return ev

    def _generate(self, question, ev, feedback=""):
        ctx = "\n\n".join(f"[{cid}] {c['text']}" for cid, c in ev.items())
        user = f"Evidence:\n{ctx}\n\nQuestion: {question}"
        if feedback:
            user += ("\n\nThese sentences in your previous draft were NOT supported by the evidence. "
                     "Fix them using only the evidence, or drop them:\n" + feedback)
        return chat(GEN_SYS, user)

    def ask(self, question, hybrid=True, use_planner=True, use_graph=True,
            use_verifier=True, max_loops=config.MAX_LOOPS):
        subs = plan(question) if use_planner else [question]
        ev = self._gather(subs, hybrid, use_graph)
        answer = self._generate(question, ev)
        claims, loops, initial, avg = [], 0, None, None

        if use_verifier:
            for loops in range(max_loops + 1):
                claims = self.v.check(answer, ev)
                if loops == 0:
                    initial = sum(c["supported"] for c in claims) / len(claims) if claims else 0.0
                bad = [c for c in claims if not c["supported"]]
                if not bad or loops == max_loops:
                    break
                for c in bad:                         # rewrite query = the failed claim itself
                    for h in self.r.search(c["text"], k=2, hybrid=hybrid):
                        ev.setdefault(h["id"], h)
                answer = self._generate(question, ev, "\n".join("- " + c["text"] for c in bad))
            good = [c["raw"] for c in claims if c["supported"]]
            grounded = len(good) / len(claims) if claims else 0.0
            avg = sum(c["score"] for c in claims) / len(claims) if claims else 0.0
            answer = " ".join(good) if good else ABSTAIN
        else:
            grounded = None

        return {"question": question, "sub_questions": subs, "answer": answer,
                "claims": claims, "grounded": grounded, "initial": initial, "avg_entail": avg, "loops": loops, "evidence": ev}
