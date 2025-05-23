from typing import List, Dict, Union
from sentence_transformers import SentenceTransformer
import numpy as np

from utils.config import EMBEDDINGS_MODEL_NAME, Colors

class EmbeddingGenerator:
    def __init__(self):
        """
        Initialize the embedding generator using environment configuration.
        """
        try:
            # Check if the model requires trusting remote code
            model_name_lower = EMBEDDINGS_MODEL_NAME.lower()
            trust_remote = "jinaai" in model_name_lower or \
                           "alibaba-nlp" in model_name_lower or \
                           "gte" in model_name_lower # GTE models from Alibaba often need this
            
            if trust_remote:
                print(f"{Colors.BLUE}Trusting remote code for model: {EMBEDDINGS_MODEL_NAME}{Colors.ENDC}")
            
            self.model = SentenceTransformer(EMBEDDINGS_MODEL_NAME, trust_remote_code=trust_remote)
            print(f"{Colors.GREEN}Successfully loaded embedding model: {EMBEDDINGS_MODEL_NAME}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}Error loading embedding model {EMBEDDINGS_MODEL_NAME}: {e}{Colors.ENDC}")
            raise
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            print(f"{Colors.RED}Warning: Empty text list provided to generate_embeddings{Colors.ENDC}")
            return []
            
        try:
            # Limit text length to avoid OOM errors
            max_length = 8192  # Adjust based on your hardware and model
            truncated_texts = [text[:max_length] for text in texts]
            
            print(f"{Colors.BLUE}Generating embeddings for {len(texts)} texts{Colors.ENDC}")
            # For BGE models, no prefix is typically needed for passage/document embeddings
            embeddings = self.model.encode(truncated_texts, show_progress_bar=True)
            
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            return embeddings
            
        except Exception as e:
            print(f"{Colors.RED}Error generating embeddings: {e}{Colors.ENDC}")
            # Return an empty list or raise the exception depending on how you want to handle errors
            raise
        
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate an embedding for a query string.
        Adds a prefix instruction if a BGE model is detected.
        
        Args:
            query: The query text
            
        Returns:
            Query embedding vector
        """
        if not query:
            raise ValueError("Empty query provided")
            
        # BGE models often recommend a prefix for query embeddings for better retrieval
        # This is a common instruction format.
        # Check if the model name (lowercase) contains 'bge' and is an English model
        query_to_embed = query
        if "bge" in EMBEDDINGS_MODEL_NAME.lower() and ("en" in EMBEDDINGS_MODEL_NAME.lower() or "icl" in EMBEDDINGS_MODEL_NAME.lower()):
            # Common instruction for BGE query embeddings.
            # Note: Some BGE variants might not strictly need this or have specific instructions.
            # Using a general one here for common BGE models.
            query_to_embed = f"Represent this sentence for searching relevant passages: {query}"
            print(f"{Colors.BLUE}Using BGE query prefix for model {EMBEDDINGS_MODEL_NAME}{Colors.ENDC}")

        try:
            embedding = self.model.encode(query_to_embed)
            return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        except Exception as e:
            print(f"{Colors.RED}Error generating query embedding: {e}{Colors.ENDC}")
            raise
