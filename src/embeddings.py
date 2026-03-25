from typing import List

import ollama

from src.config import EMBEDDINGS_MODEL_NAME, Colors


class EmbeddingGenerator:
    def __init__(self):
        self.model_name = EMBEDDINGS_MODEL_NAME
        print(f"{Colors.GREEN}Using embedding model: {self.model_name}{Colors.ENDC}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        response = ollama.embed(model=self.model_name, input=texts)
        return response['embeddings']

    def generate_query_embedding(self, query: str) -> List[float]:
        if not query:
            raise ValueError("Empty query provided")

        response = ollama.embed(model=self.model_name, input=query)
        return response['embeddings'][0]

