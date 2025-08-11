import io
import os
import logging
from openai import OpenAI
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")

class EmbeddingsService:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        # Initialize OpenAI client with the API key
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.client = OpenAI()

    def embed_text(self, text):
        embeddings= self.client.embeddings.create(input=text, model="text-embedding-ada-002")
        return embeddings
    
    def custom_embed_text(self, text):
        embeddings = model.encode([text])
        return embeddings
