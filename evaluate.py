import json
import config
from pipeline import VeriRAG
from verifier import NLIVerifier

CONFIGS = {
    "baseline (dense only)": dict(hybrid=False, use_planner=False, use_graph=False, use_verifier=False),
    "hybrid + rerank":       dict(hybrid=True,  use_planner=False, use_graph=False, use_verifier=False),
    "full VeriRAG":          dict(hybrid=True,  use_planner=True,  use_graph=True,  use_verifier=True),
}

questions = json.load(open("data/eval_questions.json"))
rag = VeriRAG()
judge = NLIVerifier(config.EVAL_NLI_MODEL)    # independent model, so we don't grade with our own verifier

results = {}
for name, flags in CONFIGS.items():
    faith, abstain, nclaims = [], 0, []
    for q in questions:
        try:
            res = rag.ask(q, **flags)
        except Exception as e:
            print("  skip:", q[:50], "|", str(e)[:80]); continue
        if res["answer"].startswith("Insufficient") or res["answer"].startswith("The evidence does not"):
            abstain += 1
            continue
        claims = judge.check(res["answer"], res["evidence"])
        if claims:
            faith.append(sum(c["supported"] for c in claims) / len(claims))
            nclaims.append(len(claims))
    results[name] = {
        "faithfulness": round(sum(faith) / len(faith), 3) if faith else None,
        "abstain_rate": round(abstain / len(questions), 3),
        "avg_claims": round(sum(nclaims) / len(nclaims), 1) if nclaims else 0,
    }
    print(name, results[name])

json.dump(results, open("results.json", "w"), indent=2)
