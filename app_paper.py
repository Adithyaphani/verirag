import gradio as gr
from paperqa import PaperQA

qa = PaperQA()


def run(paper, question, verifier):
    try:
        r = qa.ask(paper, question, use_verifier=verifier)
    except Exception as e:
        return f"Error: {e}", "", [], ""
    m = r["paper"]
    info = f"{m['title']} ({m['published']}) | {', '.join(m['authors'])} | arXiv:{m['id']}"
    rows = [[c["text"], ", ".join(c["cites"]), round(c["score"], 2),
             "supported" if c["supported"] else "unsupported"] for c in r["claims"]]
    if r["grounded"] is None:
        status = "Verifier off"
    else:
        status = (f"{r['grounded']:.0%} grounded | first draft {r['initial']:.0%} | "
                  f"avg entailment {r['avg_entail']:.2f} | {r['loops']} loop(s)")
    return r["answer"], status, rows, info


demo = gr.Interface(
    fn=run,
    inputs=[gr.Textbox(label="Paper (title, arXiv ID or URL)", value="Attention Is All You Need"),
            gr.Textbox(label="Question", lines=2),
            gr.Checkbox(value=True, label="Verifier loop")],
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Confidence"),
             gr.Dataframe(headers=["Claim", "Cited chunks", "Entailment", "Verdict"], label="Claim audit"),
             gr.Textbox(label="Paper")],
    title="VeriRAG: ask any arXiv paper, get a verified answer")

if __name__ == "__main__":
    demo.launch()
