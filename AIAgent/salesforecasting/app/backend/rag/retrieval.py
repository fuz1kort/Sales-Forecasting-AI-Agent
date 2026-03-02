import chromadb
from sentence_transformers import SentenceTransformer
import os

CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("sales_docs")
embedder = SentenceTransformer(EMBED_MODEL)

def search_chroma(query, n_results=5):
    query_emb = embedder.encode([query])
    results = collection.query(query_embeddings=query_emb, n_results=n_results)
    return [
        {
            "document": doc,
            "distance": dist
        }
        for doc, dist in zip[tuple](results['documents'][0], results['distances'][0])
    ]