from sklearn.metrics.pairwise import cosine_similarity
from src.ingest import embed

def rerank(query, candidates):
    q_emb = embed(query).reshape(1, -1)

    scored = []
    for c in candidates:
        score = cosine_similarity(q_emb, embed(c).reshape(1, -1))[0][0]
        scored.append((c, score))

    return sorted(scored, key=lambda x: x[1], reverse=True)
