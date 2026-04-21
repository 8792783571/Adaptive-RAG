from openai import OpenAI
import numpy as np

client = OpenAI(api_key="YOUR_API_KEY")

def load_docs(path="data/docs.txt"):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def embed(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding)
