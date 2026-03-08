from sentence_transformers import SentenceTransformer

class Embedder:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode_documents(self, documents):
        embeddings = self.model.encode(
            documents,
            batch_size=64,
            show_progress_bar=True
        )
        return embeddings

    def encode_query(self, query):
        return self.model.encode([query])[0]