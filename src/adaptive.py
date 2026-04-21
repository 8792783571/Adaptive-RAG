import time
from src.reranker import rerank

def adaptive_pipeline(query, retriever, generator):
    start = time.time()

    candidates, k = retriever.retrieve(query)
    latency = time.time() - start

    # Adaptive logic
    if latency > 1:
        k = max(2, k - 1)

    ranked = rerank(query, candidates)
    context = " ".join([c[0] for c in ranked[:k]])

    answer = generator(query, context)

    quality = len(answer)

    return {
        "answer": answer,
        "latency": latency,
        "k_used": k,
        "quality": quality
    }
