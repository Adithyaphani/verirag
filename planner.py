import json, re
from llm import chat

SYS = ("You split research questions into search sub-questions. Reply with ONLY a JSON list "
       "of 1-3 short strings. If the question is simple, return it as the only item.")


def plan(question):
    try:
        out = chat(SYS, question, max_tokens=200, temperature=0)
        subs = json.loads(re.search(r"\[.*\]", out, re.S).group(0))
        subs = [s for s in subs if isinstance(s, str)][:3]
        return subs or [question]
    except Exception:
        return [question]
