import os
from dotenv import load_dotenv

load_dotenv()

class Colors:
    GREEN = '\033[92m'
    GREY = '\033[90m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'

# Model configuration
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "qwen3-embedding:latest")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-oss:latest")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8000"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_TOP_K = int(os.getenv("LLM_TOP_K", "20"))
LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "0.1"))
LLM_FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.1"))
LLM_SEED = int(os.getenv("LLM_SEED", "42"))
LLM_SYSTEM_PROMPT = os.getenv("LLM_SYSTEM_PROMPT", (
    "You are a strict, fact-based assistant. Your ONLY knowledge comes from the SOURCES provided in the prompt. "
    "RULES YOU MUST FOLLOW:\n"
    "1. ONLY use information explicitly stated in the provided SOURCES. Never use outside knowledge.\n"
    "2. Quote or closely paraphrase the source text. Do not rephrase in a way that changes meaning.\n"
    "3. Cite every claim with the source number, e.g. [1].\n"
    "4. If the SOURCES do not contain enough information to answer, respond ONLY with: "
    "'I could not find this in the provided sources.'\n"
    "5. Never speculate, infer beyond what is written, or add information not present in the sources.\n"
    "6. If the question is ambiguous given the sources, state what is unclear and what the sources do say.\n"
    "7. Do not make up facts, dates, numbers, names, or any other details."
))

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
RESOLVE_REFERENCES = int(os.getenv("RESOLVE_REFERENCES", "1"))

# Guardrail configuration
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
NO_ANSWER_RESPONSE = os.getenv("NO_ANSWER_RESPONSE", "I could not find this in the provided sources.")

# Verbose output
VERBOSE = int(os.getenv("VERBOSE", "0"))
