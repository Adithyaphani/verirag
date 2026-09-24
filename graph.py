from collections import Counter
import numpy as np, networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer


def build_graph(chunks, top_terms=8):
    """Bipartite graph: chunk nodes <-> key-term nodes (TF-IDF keywords)."""
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english",
                          min_df=2, max_df=0.3, max_features=50000)
    X = vec.fit_transform([c["text"] for c in chunks]).tocsr()
    terms = vec.get_feature_names_out()
    G = nx.Graph()
    for i, c in enumerate(chunks):
        row = X[i]
        if row.nnz == 0:
            continue
        top = row.indices[np.argsort(-row.data)[:top_terms]]
        for j in top:
            G.add_edge(c["id"], "t:" + terms[j], w=float(row[0, j]))
    return G


def expand(G, chunk_ids, exclude, n=2):
    """Chunks that share key terms with the retrieved ones (2-hop walk via term nodes)."""
    scores = Counter()
    for cid in chunk_ids:
        if cid not in G:
            continue
        for term in G[cid]:
            for nb in G[term]:
                if nb != cid and nb not in exclude:
                    scores[nb] += G[cid][term]["w"] * G[term][nb]["w"]
    return [c for c, _ in scores.most_common(n)]
