import time
import json
from src.reranker import rerank

def save_metrics(data, path="results/metrics.json"):
    try:
        with open(path, "r") as f:
            existing = json.load(f)
    except:
        existing = []

    existing.append(data)

    with open(path, "w") as f:
        json.dump(existing, f, indent=4)


def adaptive_pipeline(query, retriever, generator):
    start = time.time()

    candidates, k = retriever.retrieve(query)
    latency = time.time() - start

    if latency > 1:
        k = max(2, k - 1)

    ranked = rerank(query, candidates)
    context = " ".join([c[0] for c in ranked[:k]])

    answer = generator(query, context)
    quality = len(answer)

    result = {
        "query": query,
        "latency": latency,
        "k_used": k,
        "quality": quality
    }

    # 🔥 Save metrics
    save_metrics(result)

    return result
