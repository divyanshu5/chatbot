import chromadb
from chromadb.config import Settings

class ChromaDB:
    def __init__(self, persist_directory="./chroma_db", collection_name="pdf_chunks"):
        self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_embeddings(self, texts, embeddings):
        ids = [f"chunk-{i}" for i in range(len(texts))]
        self.collection.add(
            embeddings=[e.tolist() if hasattr(e, 'tolist') else e for e in embeddings],
            documents=texts,
            ids=ids
        )

    def query(self, query_embedding, n_results=5):
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "distances", "ids"]
        )
        return results

    def clear(self):
        self.collection.delete(where={})

    def get_all(self):
        # Returns all documents and ids in the collection
        return self.collection.get(include=["documents", "ids"])