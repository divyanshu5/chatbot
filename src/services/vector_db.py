import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension of the embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        self.text_chunks = []
        logger.debug("VectorDB initialized")

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to split
            chunk_size: Size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks

    def add_documents(self, text: str):
        """
        Add documents to the vector database.
        
        Args:
            text: The text to add
        """
        try:
            # Split text into chunks
            chunks = self.chunk_text(text)
            self.text_chunks.extend(chunks)
            
            # Generate embeddings for chunks
            embeddings = self.model.encode(chunks)
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            logger.debug(f"Added {len(chunks)} chunks to vector database")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}", exc_info=True)
            raise

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for similar chunks in the vector database.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of (chunk, distance) tuples
        """
        try:
            # Generate embedding for query
            query_embedding = self.model.encode([query])[0].astype('float32')
            
            # Search in FAISS
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
            
            # Get the chunks and their distances
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # -1 indicates no result
                    results.append((self.text_chunks[idx], float(distances[0][i])))
            
            logger.debug(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}", exc_info=True)
            raise

    def save_index(self, path: str = "vector_db.index"):
        """Save the FAISS index to disk."""
        try:
            faiss.write_index(self.index, path)
            logger.debug(f"Saved index to {path}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}", exc_info=True)
            raise

    def load_index(self, path: str = "vector_db.index"):
        """Load the FAISS index from disk."""
        try:
            self.index = faiss.read_index(path)
            logger.debug(f"Loaded index from {path}")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}", exc_info=True)
            raise 