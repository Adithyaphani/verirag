import gradio as gr
from pipeline import VeriRAG

rag = VeriRAG()


def run(q, hybrid, planner, graph, verifier):
    r = rag.ask(q, hybrid=hybrid, use_planner=planner, use_graph=graph, use_verifier=verifier)
    rows = [[c["text"], ", ".join(c["cites"]), round(c["score"], 2),
             "supported" if c["supported"] else "unsupported"] for c in r["claims"]]
    status = (f"{r['grounded']:.0%} grounded after {r['loops']} correction loop(s)"
              if r["grounded"] is not None else "Verifier off")
    return r["answer"], status, rows, "\n".join(r["sub_questions"])


demo = gr.Interface(
    fn=run,
    inputs=[gr.Textbox(label="Question", lines=2),
            gr.Checkbox(True, label="Hybrid + reranker"), gr.Checkbox(True, label="Planner"),
            gr.Checkbox(True, label="Concept graph"), gr.Checkbox(True, label="Verifier loop")],
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Confidence"),
             gr.Dataframe(headers=["Claim", "Cited chunks", "Entailment", "Verdict"], label="Claim audit"),
             gr.Textbox(label="Planner sub-questions")],
    title="VeriRAG: self-auditing research agent (quantum error correction)")

if __name__ == "__main__":
    demo.launch()
