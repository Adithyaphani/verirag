import json, os, re, time, requests, arxiv
from pypdf import PdfReader

CHUNK_WORDS, OVERLAP = 250, 50
QUERY, N = "quantum error correction", 30
os.makedirs("papers", exist_ok=True)

chunks = []
search = arxiv.Search(query=QUERY, max_results=N, sort_by=arxiv.SortCriterion.Relevance)
for r in arxiv.Client().results(search):
    pid = r.get_short_id().replace("/", "_")
    path = f"papers/{pid}.pdf"
    try:
        if not os.path.exists(path):
            resp = requests.get(r.pdf_url, timeout=60, headers={"User-Agent": "verirag-research"})
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
            time.sleep(3)
        text = " ".join(p.extract_text() or "" for p in PdfReader(path).pages)
    except Exception as e:
        print("skip", pid, e); continue
    words = re.sub(r"\s+", " ", text).split()
    step = CHUNK_WORDS - OVERLAP
    for i in range(0, max(len(words) - OVERLAP, 1), step):
        chunks.append({"id": f"{pid}_{i//step}", "paper": pid, "title": r.title,
                       "text": " ".join(words[i:i + CHUNK_WORDS])})
    print(pid, r.title[:60], len(words), "words")

with open("chunks.jsonl", "w") as f:
    for c in chunks:
        f.write(json.dumps(c) + "\n")
print("total chunks:", len(chunks))
