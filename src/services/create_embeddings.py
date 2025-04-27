from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class CreateEmbeddings:
    def __init__(self):
        load_dotenv()
        self.embeddings = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def create_embeddings(self, text):
        self.embeddings = []
        self.embeddings = self.model.encode(text)
        print(self.embeddings.shape)
        
        # Save embeddings to file
        np.save('embeddings.npy', self.embeddings)
        print(f"Embeddings saved to embeddings.npy")
        
        return self.embeddings

    def get_embeddings(self):
        return self.embeddings

