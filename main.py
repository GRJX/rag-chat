import argparse
import uuid
from typing import List, Dict, Any, Optional, Generator, Union, Tuple
from pathlib import Path

from utils.indexer import Indexer
from utils.embeddings import EmbeddingGenerator
from utils.db_handler import ChromaDBHandler
from utils.generator import OllamaGenerator
from utils.config import N_RESULTS, VERBOSE, Colors, SIMILARITY_THRESHOLD, MIN_CHUNK_SIZE, ENABLE_RERANKING

def index_data(directory: str, db_handler: ChromaDBHandler, embedding_generator: EmbeddingGenerator) -> None:
    """
    Index a codebase and store it in the database.
    
    Args:
        directory: Directory containing the codebase
        db_handler: ChromaDB handler instance
        embedding_generator: Embedding generator instance
    """
    indexer = Indexer()
    
    # Automatically detect supported file types in the directory
    supported_extensions = indexer.get_supported_file_types()
    discovered_files = discover_files_by_type(directory, supported_extensions)
    
    if not discovered_files:
        print(f"{Colors.RED}No supported files found in {directory}{Colors.ENDC}")
        return
        
    # Print summary of discovered files
    for ext, count in discovered_files.items():
        print(f"{Colors.GREEN}Found {count} {ext} files{Colors.ENDC}")
    
    documents = []
    metadatas = []
    ids = []
    
    print(f"{Colors.BLUE}Traversing directory: {directory}{Colors.ENDC}")
    
    # Process all file types with automatic handling
    for file_path, content in indexer.traverse_directory(directory, auto_detect=True):
        print(f"{Colors.BLUE}Processing file: {file_path}{Colors.ENDC}")
        chunks = indexer.chunk_data(file_path, content)
        
        # Prepare data for batch insertion
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            documents.append(chunk['content'])
            metadatas.append({
                'file_path': chunk['file_path'],
                'start_line': chunk['start_line'],
                'end_line': chunk['end_line']
            })
            ids.append(chunk_id)
    
    if not documents:
        print(f"{Colors.RED}No documents found to index. Please check the directory path.{Colors.ENDC}")
        return
        
    print(f"{Colors.BLUE}Generating embeddings for {len(documents)} chunks...{Colors.ENDC}")
    # Process in batches to avoid memory issues
    batch_size = 100  # Adjust based on your hardware capabilities
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(documents))
        
        print(f"{Colors.BLUE}Processing batch {i+1}/{total_batches} (documents {start_idx}-{end_idx})...{Colors.ENDC}")
        
        batch_documents = documents[start_idx:end_idx]
        batch_metadatas = metadatas[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]
        
        try:
            batch_embeddings = embedding_generator.generate_embeddings(batch_documents)
            
            if not batch_embeddings or len(batch_embeddings) == 0:
                print(f"{Colors.RED}Warning: Batch {i+1} produced empty embeddings. Skipping this batch.{Colors.ENDC}")
                continue
                
            if len(batch_embeddings) != len(batch_documents):
                print(f"{Colors.RED}Warning: Mismatch between documents ({len(batch_documents)}) and embeddings ({len(batch_embeddings)}). Adjusting...{Colors.ENDC}")
                # Ensure equal length by truncating to the shorter length
                min_len = min(len(batch_documents), len(batch_embeddings))
                batch_documents = batch_documents[:min_len]
                batch_embeddings = batch_embeddings[:min_len]
                batch_metadatas = batch_metadatas[:min_len]
                batch_ids = batch_ids[:min_len]
            
            print(f"{Colors.BLUE}Adding batch {i+1} to the database ({len(batch_embeddings)} documents)...{Colors.ENDC}")
            db_handler.add_documents(batch_documents, batch_embeddings, batch_metadatas, batch_ids)
            
        except Exception as e:
            print(f"{Colors.RED}Error processing batch {i+1}: {e}{Colors.ENDC}")
            continue
    
    print(f"{Colors.GREEN}Indexed {db_handler.collection_count()} chunks successfully{Colors.ENDC}")

def discover_files_by_type(directory: str, supported_extensions: List[str]) -> Dict[str, int]:
    """
    Discover files by type in a directory.
    
    Args:
        directory: Directory to search
        supported_extensions: List of supported file extensions
        
    Returns:
        Dictionary of file extensions and counts
    """
    result = {}
    dir_path = Path(directory)
    
    for ext in supported_extensions:
        files = list(dir_path.rglob(f"*{ext}"))
        if files:
            result[ext] = len(files)
    
    return result

def query_codebase(query: str, db_handler: ChromaDBHandler, embedding_generator: EmbeddingGenerator, 
                  generator: OllamaGenerator, chat_history: Optional[List[Dict[str, str]]] = None,
                  initial_topic: Optional[str] = None, is_followup: bool = False,
                  cached_contexts: Optional[List[Dict[str, Any]]] = None) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
    """
    Query the codebase and generate a streamed response, maintaining context of the initial topic.
    For follow-up questions, skips the RAG search and reuses the initial contexts.
    
    Args:
        query: The user's current query
        db_handler: ChromaDB handler instance
        embedding_generator: Embedding generator instance
        generator: Ollama generator instance
        chat_history: Optional list of previous chat messages
        initial_topic: The initial query/topic to maintain focus on
        is_followup: Whether this is a follow-up question (skips RAG search)
        cached_contexts: Previously retrieved contexts to reuse for follow-up questions
            
    Returns:
        Tuple of (response_generator, contexts) where response_generator yields response chunks
        and contexts is the list of context chunks used for RAG
    """
    print(f"{Colors.BLUE}Query: {query}{Colors.ENDC}")
    
    contexts = cached_contexts or []
    
    # Only perform RAG search for the initial query, not for follow-ups
    if not is_followup:
        print(f"{Colors.BLUE}Generating query embedding...{Colors.ENDC}")
        query_embedding = embedding_generator.generate_query_embedding(query)
        
        print(f"{Colors.BLUE}Searching for top {N_RESULTS} relevant chunks with quality filtering...{Colors.ENDC}")
        try:
            # Use enhanced query with similarity filtering
            results = db_handler.query(
                query_embedding, 
                n_results=N_RESULTS,
                similarity_threshold=SIMILARITY_THRESHOLD,
                min_chunk_size=MIN_CHUNK_SIZE,
                rerank_by_section=bool(ENABLE_RERANKING),
                prefer_headers=bool(ENABLE_RERANKING)
            )
        except Exception as e:
            print(f"{Colors.RED}Unexpected error during query: {e}{Colors.ENDC}")
            return ((_ for _ in [f"Error during query: {e}"]), [])
        
        contexts = []
        if results and results.get('documents') and results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                # Use the database handler's similarity conversion
                similarity = db_handler._convert_distance_to_similarity(distance)
                
                print(f"{Colors.GREEN}Match {i+1}: {metadata['file_path']} (Lines {metadata['start_line']}-{metadata['end_line']}) - Similarity: {similarity:.4f} (Distance: {distance:.4f}){Colors.ENDC}")
                
                # If verbose mode is enabled, print the chunk content
                if VERBOSE == 1:
                    print(f"{Colors.GREY}\n--- Chunk Content ---")
                    print(f"\t{doc}")
                    print(f"-------------------{Colors.ENDC}\n")
                    
                contexts.append({
                    'content': doc,
                    'file_path': metadata['file_path'],
                    'start_line': metadata['start_line'],
                    'end_line': metadata['end_line'],
                })
        else:
            print(f"{Colors.YELLOW}No high-quality chunks found for the query. Try lowering SIMILARITY_THRESHOLD.{Colors.ENDC}")
    else:
        print(f"{Colors.BLUE}Using cached contexts from initial query \"{initial_topic}\" (skipping RAG search)...{Colors.ENDC}")

    print(f"{Colors.BLUE}\nConstructing prompt with context...{Colors.ENDC}")
    
    # Pass the initial topic to keep focus
    prompt = generator.construct_prompt(query, contexts, chat_history, initial_topic)
    
    # If verbose mode is enabled, print the full prompt
    if VERBOSE == 1:
        print(f"{Colors.GREY}\n--- Full Prompt ---")
        print(prompt)  # Print full prompt
        print(f"-------------------{Colors.ENDC}\n")
    
    print(f"{Colors.BLUE}\nGenerating streamed response...\n{Colors.ENDC}")
    
    # Return both the response generator and the contexts
    return (generator.generate(prompt, stream=True), contexts)

def main():
    parser = argparse.ArgumentParser(description="Local RAG system for code understanding")
    parser.add_argument("--index", type=str, help="Directory to index")
    parser.add_argument("--query", type=str, help="Initial query to start the chat")
    args = parser.parse_args()
    
    # Initialize components using only environment configuration
    embedding_generator = EmbeddingGenerator()
    db_handler = ChromaDBHandler()
    generator = OllamaGenerator()
    
    if args.index:
        index_data(args.index, db_handler, embedding_generator)
    else:
        if db_handler.collection_count() == 0:
            print(f"{Colors.RED}No documents indexed yet. Please index a codebase first with --index <directory>.{Colors.ENDC}")
            return
        
        if not args.query:
            print(f"{Colors.RED}Please provide an initial query with --query to start a focused chat.{Colors.ENDC}")
            return
            
        # Start a focused chat session
        print(f"{Colors.GREEN}Starting focused chat session with {generator.model_name}.{Colors.ENDC}")
        print(f"{Colors.GREEN}Initial topic: \"{args.query}\"{Colors.ENDC}")
        print(f"{Colors.YELLOW}Note: Only the initial query uses RAG search. Follow-up questions will reuse the same context.{Colors.ENDC}")
        print(f"{Colors.GREEN}Type 'exit', 'quit', or press Ctrl+C to end the session.{Colors.ENDC}")
        
        # Store the initial topic to maintain focus
        initial_topic = args.query
        chat_history = []
        
        # Process the initial query
        print(f"{Colors.BLUE}You: {args.query}{Colors.ENDC}")
        
        # Generate streaming response for initial query and collect retrieved contexts
        # The query_codebase function now returns both the response generator and the contexts
        print(f"{Colors.GREEN}Assistant:{Colors.ENDC} ", end="", flush=True)
        full_response = ""
        response_stream, cached_contexts = query_codebase(args.query, db_handler, embedding_generator, generator, None, initial_topic)
            
        # Stream the response
        for chunk in response_stream:
            print(chunk, end="", flush=True)
            full_response += chunk
        print()  # Add newline after response
        
        # Add to chat history
        chat_history.append({"role": "user", "content": args.query})
        chat_history.append({"role": "assistant", "content": full_response})
        
        # Start interactive loop
        while True:
            try:
                user_input = input(f"\n{Colors.BLUE}You: {Colors.ENDC}")
                
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting chat.")
                    break
                
                if not user_input.strip():
                    continue
                
                # Generate streamed response maintaining context of the initial topic
                # But skip the RAG search by setting is_followup=True
                print(f"{Colors.GREEN}Assistant:{Colors.ENDC} ", end="", flush=True)
                full_response = ""
                response_stream, _ = query_codebase(user_input, db_handler, embedding_generator, generator, 
                                                  chat_history, initial_topic, is_followup=True, 
                                                  cached_contexts=cached_contexts)
                
                for chunk in response_stream:
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()  # Add newline after response
                
                # Update chat history
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": full_response})
                
            except KeyboardInterrupt:
                print("\nExiting chat.")
                break

if __name__ == "__main__":
    main()
