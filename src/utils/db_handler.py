import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Any

from utils.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY, Colors

class ChromaDBHandler:
    def __init__(self):
        """
        Initialize the ChromaDB handler using environment configuration.
        """
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        
        # Create or get the collection
        try:
            self.collection = self.client.get_collection(name=CHROMA_COLLECTION_NAME)
            print(f"{Colors.GREEN}Using existing collection: {CHROMA_COLLECTION_NAME}{Colors.ENDC}")
        except:
            print(f"{Colors.BLUE}Creating new collection: {CHROMA_COLLECTION_NAME}{Colors.ENDC}")
            self.collection = self.client.create_collection(name=CHROMA_COLLECTION_NAME)
    
    def add_documents(self, 
                     documents: List[str], 
                     embeddings: List[List[float]], 
                     metadatas: List[Dict[str, Any]], 
                     ids: List[str]) -> None:
        """
        Add documents to the collection.
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: List of document IDs
        """
        if not documents or not embeddings:
            raise ValueError("Cannot add empty documents or embeddings to the collection")
            
        if len(documents) != len(embeddings) or len(documents) != len(metadatas) or len(documents) != len(ids):
            raise ValueError(f"Mismatch in lengths: documents ({len(documents)}), embeddings ({len(embeddings)}), "
                           f"metadatas ({len(metadatas)}), ids ({len(ids)})")
        
        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"{Colors.GREEN}Successfully added {len(documents)} documents to collection{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.RED}Error adding documents to collection: {e}{Colors.ENDC}")
            raise
    
    def query(self, 
              query_embedding: List[float], 
              n_results: int = 5) -> Dict[str, Any]:
        """
        Query the collection using a query embedding.
        
        Args:
            query_embedding: The query embedding vector
            n_results: Number of results to return
            
        Returns:
            Query results with documents, metadatas, and distances
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    
    def collection_count(self) -> int:
        """
        Get the number of items in the collection.
        
        Returns:
            Count of items
        """
        return self.collection.count()
