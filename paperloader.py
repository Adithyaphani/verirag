import os, re, time, requests, arxiv
from pypdf import PdfReader
from textclean import clean

CHUNK_WORDS, OVERLAP = 250, 50
NEW_ID = r"\d{4}\.\d{4,5}"
OLD_ID = r"[a-z\-]+(?:\.[A-Za-z]{2})?/\d{7}"
ID = re.compile(rf"^(?:{NEW_ID}|{OLD_ID})(?:v\d+)?$")
URL = re.compile(rf"arxiv\.org/(?:abs|pdf)/((?:{NEW_ID}|{OLD_ID})(?:v\d+)?)")


def find_paper(query):
    q = query.strip()
    m = URL.search(q)
    if m:
        q = m.group(1)
    client = arxiv.Client(num_retries=5, delay_seconds=3)
    if ID.match(q):
        search = arxiv.Search(id_list=[q])
    else:
        search = arxiv.Search(query=f'ti:"{q}"', max_results=1)
    r = next(client.results(search), None)
    if r is None and not ID.match(q):           # fall back to a looser search
        r = next(client.results(arxiv.Search(query=q, max_results=1)), None)
    return r


def load_paper(query):
    r = find_paper(query)
    if r is None:
        raise ValueError(f"No arXiv paper found for: {query}")
    pid = r.get_short_id().replace("/", "_")
    os.makedirs("papers", exist_ok=True)
    path = f"papers/{pid}.pdf"
    if not os.path.exists(path):
        resp = requests.get(r.pdf_url, timeout=60, headers={"User-Agent": "verirag-research"})
        resp.raise_for_status()
        open(path, "wb").write(resp.content)
        time.sleep(3)                            # be polite to arXiv
    text = " ".join(p.extract_text() or "" for p in PdfReader(path).pages)
    words = clean(text).split()
    step = CHUNK_WORDS - OVERLAP
    chunks = [{"id": f"{pid}_{i // step}", "paper": pid, "title": r.title,
               "text": " ".join(words[i:i + CHUNK_WORDS])}
              for i in range(0, max(len(words) - OVERLAP, 1), step)]
    meta = {"id": pid, "title": r.title, "authors": [a.name for a in r.authors][:5],
            "published": str(r.published.date()), "abstract": r.summary}
    return meta, chunks
