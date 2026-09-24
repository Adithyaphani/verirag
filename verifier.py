import re, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import config

CITE = re.compile(r"\[([^\[\]]+?_\d+)\]")


def split_claims(answer):
    out = []
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", answer.strip()):
        s = s.strip()
        if len(s.split()) < 4:
            continue
        out.append({"raw": s, "text": CITE.sub("", s).replace(" .", ".").strip(),
                    "cites": CITE.findall(s)})
    return out


class NLIVerifier:
    def __init__(self, model_name=config.NLI_MODEL):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.ent = [i for i, l in self.model.config.id2label.items()
                    if l.lower().startswith("entail")][0]

    @torch.no_grad()
    def entail(self, premise, claim):
        x = self.tok(premise, claim, truncation="only_first", max_length=512, return_tensors="pt")
        return float(torch.softmax(self.model(**x).logits, -1)[0][self.ent])

    def check(self, answer, by_id):
        """Score each claim against its cited chunks (or all evidence if uncited/invalid cites)."""
        claims = split_claims(answer)
        for c in claims:
            ids = [i for i in c["cites"] if i in by_id] or list(by_id)
            c["score"] = max((self.entail(by_id[i]["text"], c["text"]) for i in ids), default=0.0)
            c["supported"] = c["score"] >= config.ENTAIL_THRESHOLD
        return claims
