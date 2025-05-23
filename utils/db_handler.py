import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Any, Union

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
            
            # Try to determine the distance function being used
            self._detect_distance_function()
            
        except:
            print(f"{Colors.BLUE}Creating new collection: {CHROMA_COLLECTION_NAME}{Colors.ENDC}")
            # Create collection with explicit distance function (cosine similarity)
            self.collection = self.client.create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}  # Use cosine distance
            )
            self.distance_function = "cosine"
            print(f"{Colors.BLUE}Created collection with cosine distance function{Colors.ENDC}")

    def _detect_distance_function(self):
        """Detect which distance function the collection is using."""
        try:
            metadata = self.collection.metadata
            if metadata and "hnsw:space" in metadata:
                self.distance_function = metadata["hnsw:space"]
                print(f"{Colors.BLUE}Detected distance function: {self.distance_function}{Colors.ENDC}")
            else:
                # Default assumption - but this might be wrong
                self.distance_function = "l2"  # Euclidean is ChromaDB default
                print(f"{Colors.YELLOW}Distance function not specified in metadata, assuming: {self.distance_function}{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.YELLOW}Could not detect distance function: {e}, assuming l2{Colors.ENDC}")
            self.distance_function = "l2"

    def _convert_distance_to_similarity(self, distance: float) -> float:
        """Convert distance to similarity score based on the distance function."""
        if self.distance_function == "cosine":
            # Cosine distance: 0 = identical, 2 = opposite
            # Similarity: 1 = identical, 0 = opposite
            return max(0.0, 1.0 - (distance / 2.0))
        
        elif self.distance_function == "l2":
            # Euclidean/L2 distance: 0 = identical, higher = more different
            # Convert to similarity using exponential decay
            return 1.0 / (1.0 + distance)
        
        elif self.distance_function == "ip":
            # Inner product: higher = more similar (but can be negative)
            # This is tricky as it depends on vector normalization
            if distance > 0:
                return min(1.0, distance)
            else:
                return 0.0
        
        else:
            # Unknown distance function - use exponential decay as fallback
            return 1.0 / (1.0 + abs(distance))

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
              n_results: int = 5,
              similarity_threshold: Optional[float] = None,
              metadata_filter: Optional[Dict[str, Any]] = None,
              file_extensions: Optional[List[str]] = None,
              file_paths: Optional[List[str]] = None,
              min_chunk_size: Optional[int] = None,
              rerank_by_section: bool = False,
              prefer_headers: bool = False) -> Dict[str, Any]:
        """
        Enhanced query with similarity filtering and metadata constraints.
        
        Args:
            query_embedding: The query embedding vector
            n_results: Number of results to return (before filtering)
            similarity_threshold: Minimum similarity score (1 - distance). Results with lower 
                                 similarity will be filtered out. Range: 0.0 to 1.0
            metadata_filter: Dictionary for filtering by metadata fields
            file_extensions: List of file extensions to filter by (e.g., ['.py', '.js'])
            file_paths: List of specific file paths to search within
            min_chunk_size: Minimum chunk size to include in results
            rerank_by_section: Whether to rerank results by section context
            prefer_headers: Whether to boost chunks that contain headers
            
        Returns:
            Filtered and potentially reranked query results
        """
        try:
            # Build metadata filter
            where_filter = {}
            
            if metadata_filter:
                where_filter.update(metadata_filter)
            
            if file_extensions:
                # Create OR condition for multiple extensions
                if len(file_extensions) == 1:
                    where_filter["$and"] = where_filter.get("$and", [])
                    where_filter["$and"].append({
                        "file_path": {"$regex": f"\\{file_extensions[0]}$"}
                    })
                else:
                    ext_conditions = []
                    for ext in file_extensions:
                        ext_conditions.append({"file_path": {"$regex": f"\\{ext}$"}})
                    where_filter["$or"] = ext_conditions
            
            if file_paths:
                # Filter by specific file paths
                if len(file_paths) == 1:
                    where_filter["file_path"] = {"$eq": file_paths[0]}
                else:
                    where_filter["file_path"] = {"$in": file_paths}
            
            # Increase n_results for initial query if we're going to filter
            initial_n_results = n_results
            if similarity_threshold is not None or min_chunk_size is not None:
                initial_n_results = min(n_results * 3, 50)  # Get more results to filter from
            
            # Perform the query
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": initial_n_results
            }
            
            if where_filter:
                query_params["where"] = where_filter
            
            results = self.collection.query(**query_params)
            
            if not results or not results.get('documents') or not results['documents'][0]:
                print(f"{Colors.YELLOW}No results found matching the criteria{Colors.ENDC}")
                return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
            
            # Apply post-query filtering
            filtered_results = self._apply_filters(
                results, 
                similarity_threshold=similarity_threshold,
                min_chunk_size=min_chunk_size,
                n_results=n_results
            )
            
            # Apply reranking if requested
            if rerank_by_section or prefer_headers:
                filtered_results = self._rerank_results(
                    filtered_results,
                    rerank_by_section=rerank_by_section,
                    prefer_headers=prefer_headers
                )
            
            # Log filtering results
            original_count = len(results['documents'][0]) if results['documents'] else 0
            final_count = len(filtered_results['documents'][0]) if filtered_results['documents'] else 0
            
            if similarity_threshold is not None or min_chunk_size is not None:
                print(f"{Colors.BLUE}Filtered {original_count} results down to {final_count} high-quality matches{Colors.ENDC}")
            
            return filtered_results
            
        except Exception as e:
            print(f"{Colors.RED}Error during enhanced query: {e}{Colors.ENDC}")
            # Fallback to basic query
            try:
                return self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
            except Exception as fallback_error:
                print(f"{Colors.RED}Fallback query also failed: {fallback_error}{Colors.ENDC}")
                return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
    
    def _apply_filters(self, 
                      results: Dict[str, Any], 
                      similarity_threshold: Optional[float] = None,
                      min_chunk_size: Optional[int] = None,
                      n_results: int = 5) -> Dict[str, Any]:
        """Apply post-query filters to results."""
        if not results or not results.get('documents') or not results['documents'][0]:
            return results
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        ids = results['ids'][0]
        
        filtered_docs = []
        filtered_metas = []
        filtered_distances = []
        filtered_ids = []
        
        # Debug: Print distance statistics
        if distances:
            min_dist = min(distances)
            max_dist = max(distances)
            avg_dist = sum(distances) / len(distances)
            print(f"{Colors.BLUE}Distance stats - Min: {min_dist:.3f}, Max: {max_dist:.3f}, Avg: {avg_dist:.3f}, Function: {getattr(self, 'distance_function', 'unknown')}{Colors.ENDC}")
        
        for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids):
            # Apply similarity threshold with proper distance handling
            if similarity_threshold is not None:
                similarity = self._convert_distance_to_similarity(dist)
                
                if similarity < similarity_threshold:
                    print(f"{Colors.GREY}Filtered out result with similarity {similarity:.3f} (distance: {dist:.3f}) < {similarity_threshold}{Colors.ENDC}")
                    continue
            
            # Apply minimum chunk size filter
            if min_chunk_size is not None:
                if len(doc) < min_chunk_size:
                    print(f"{Colors.GREY}Filtered out small chunk ({len(doc)} chars < {min_chunk_size}){Colors.ENDC}")
                    continue
            
            filtered_docs.append(doc)
            filtered_metas.append(meta)
            filtered_distances.append(dist)
            filtered_ids.append(doc_id)
            
            # Stop if we have enough results
            if len(filtered_docs) >= n_results:
                break
        
        return {
            "documents": [filtered_docs],
            "metadatas": [filtered_metas], 
            "distances": [filtered_distances],
            "ids": [filtered_ids]
        }
    
    def _rerank_results(self, 
                       results: Dict[str, Any],
                       rerank_by_section: bool = False,
                       prefer_headers: bool = False) -> Dict[str, Any]:
        """Rerank results based on additional criteria."""
        if not results or not results.get('documents') or not results['documents'][0]:
            return results
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        ids = results['ids'][0]
        
        # Create tuples for reranking
        result_tuples = list(zip(documents, metadatas, distances, ids))
        
        def rerank_score(item):
            doc, meta, dist, doc_id = item
            score = -dist  # Start with negative distance (higher similarity = higher score)
            
            # Boost for headers
            if prefer_headers:
                if any(line.strip().startswith('#') for line in doc.split('\n')[:3]):
                    score += 0.1  # Boost header chunks
                
                # Check for section context in metadata
                if meta.get('section_context'):
                    score += 0.05  # Boost chunks with section context
            
            # Boost for section coherence
            if rerank_by_section:
                # Prefer chunks from the same file as top results
                if result_tuples and meta.get('file_path') == result_tuples[0][1].get('file_path'):
                    score += 0.02
                
                # Prefer chunks with more structured content
                lines = doc.split('\n')
                if any(line.strip().startswith(('def ', 'class ', 'function ', '##')) for line in lines):
                    score += 0.03
            
            return score
        
        # Sort by rerank score
        result_tuples.sort(key=rerank_score, reverse=True)
        
        # Unpack back to separate lists
        reranked_docs, reranked_metas, reranked_distances, reranked_ids = zip(*result_tuples) if result_tuples else ([], [], [], [])
        
        return {
            "documents": [list(reranked_docs)],
            "metadatas": [list(reranked_metas)],
            "distances": [list(reranked_distances)],
            "ids": [list(reranked_ids)]
        }
    
    def query_by_file_type(self, 
                          query_embedding: List[float],
                          file_extension: str,
                          n_results: int = 5,
                          similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Convenience method to query only specific file types with high similarity.
        
        Args:
            query_embedding: The query embedding vector
            file_extension: File extension to filter by (e.g., '.py')
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            Query results filtered by file type and similarity
        """
        return self.query(
            query_embedding=query_embedding,
            n_results=n_results,
            similarity_threshold=similarity_threshold,
            file_extensions=[file_extension],
            prefer_headers=True
        )
    
    def query_high_quality(self,
                          query_embedding: List[float],
                          n_results: int = 5,
                          similarity_threshold: float = 0.75,
                          min_chunk_size: int = 100) -> Dict[str, Any]:
        """
        Convenience method for high-quality results only.
        
        Args:
            query_embedding: The query embedding vector
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score (default 0.75 for high quality)
            min_chunk_size: Minimum chunk size to include
            
        Returns:
            High-quality query results
        """
        return self.query(
            query_embedding=query_embedding,
            n_results=n_results,
            similarity_threshold=similarity_threshold,
            min_chunk_size=min_chunk_size,
            rerank_by_section=True,
            prefer_headers=True
        )
    
    def collection_count(self) -> int:
        """
        Get the number of items in the collection.
        
        Returns:
            Count of items
        """
        return self.collection.count()
