import os
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Color codes
class Colors:
    GREEN = '\033[92m'
    GREY = '\033[90m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'

# Model configuration
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "microsoft/unixcoder-base")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemma3:12b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.5"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))
LLM_SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT", "You are a helpful programming assistant.")

# Indexing configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# ChromaDB configuration
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "code_collection")

# Retrieval configuration
N_RESULTS = int(os.getenv("N_RESULTS", 5))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "50"))
ENABLE_RERANKING = int(os.getenv("ENABLE_RERANKING", "1"))

# Verbose output
VERBOSE = int(os.getenv("VERBOSE", "0"))  # Convert string to int (0 or 1)

# Parse supported extensions
def get_supported_extensions() -> Dict[str, str]:
    """Get supported file extensions from environment variables"""
    extensions_str = os.getenv("SUPPORTED_EXTENSIONS", ".py,.js,.java,.html,.css,.md,.txt,.pdf")
    extensions = [ext.strip() for ext in extensions_str.split(",")]
    
    # Default descriptions for known extensions
    descriptions = {
        '.py': 'Python code',
        '.js': 'JavaScript code',
        '.java': 'Java code',
        '.html': 'HTML markup',
        '.css': 'CSS stylesheet',
        '.md': 'Markdown text',
        '.txt': 'Plain text',
        '.pdf': 'PDF document',
    }
    
    # Create a dictionary with available descriptions or generic ones
    result = {}
    for ext in extensions:
        if ext in descriptions:
            result[ext] = descriptions[ext]
        else:
            result[ext] = f'{ext[1:].upper()} file'
    
    return result

SUPPORTED_EXTENSIONS = get_supported_extensions()
