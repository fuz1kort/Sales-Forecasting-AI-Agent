# import chromadb
# from sentence_transformers import SentenceTransformer
# import os
#
# CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
# EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
#
# client = chromadb.PersistentClient(path=CHROMA_DIR)
# collection = client.get_or_create_collection("sales_docs")
# embedder = SentenceTransformer(EMBED_MODEL)
#
# def index_embeddings(df, text_column):
#     docs = df[text_column].astype(str).tolist()
#     embeddings = embedder.encode(docs)
#     ids = [f"doc_{i}" for i in range(len(docs))]
#     collection.add(documents=docs, embeddings=embeddings, ids=ids)
#     return len(docs)