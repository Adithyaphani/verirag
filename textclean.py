import re

LIG = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}


def clean(t):
    for k, v in LIG.items():
        t = t.replace(k, v)
    t = re.sub(r"(\w)- (\w)", r"\1\2", t)                    # "em- bedded" -> "embedded"
    t = re.sub(r"\[\d+(?:[,\u2013-]\s*\d+)*\]", "", t)       # paper's own refs like [39]
    return re.sub(r"\s+", " ", t).strip()
