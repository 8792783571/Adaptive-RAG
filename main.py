from src.ingest import load_docs
from src.retriever import Retriever
from src.generator import generate
from src.adaptive import adaptive_pipeline

docs = load_docs()
retriever = Retriever(docs)

query = "What is FAISS?"

result = adaptive_pipeline(query, retriever, generate)

print(result)
