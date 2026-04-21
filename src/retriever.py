import faiss
import numpy as np
from src.ingest import embed

class Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.embeddings = np.array([embed(d) for d in docs])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def dynamic_k(self, query):
        return 2 if len(query.split()) < 5 else 5

    def keyword_filter(self, query):
        return [d for d in self.docs if any(w in d.lower() for w in query.lower().split())]

    def retrieve(self, query):
        k = self.dynamic_k(query)
        q_emb = embed(query).reshape(1, -1)

        _, idx = self.index.search(q_emb, k)
        vector_results = [self.docs[i] for i in idx[0]]

        keyword_results = self.keyword_filter(query)

        return list(set(vector_results + keyword_results)), k
