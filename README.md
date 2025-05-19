# Code-RAG: Local Retrieval-Augmented Generation for Code Understanding

A local RAG (Retrieval-Augmented Generation) system that helps you understand and query your codebase using locally-run language models through a focused interactive chat interface.

## Requirements

- Python 3.8+
- Ollama installed and running locally
- Ensure your environment is configured with the desired embedding model (default is `BAAI/bge-en-icl` as per `.env` file).
  - Note: Some models, like those from JinaAI or Alibaba-NLP (e.g., GTE models), may require `trust_remote_code=True` to load. The system attempts to handle this automatically for known cases. Be aware of the security implications of running remote code.

## Installation

1. Clone this repository:

2. Install dependencies:
```bash
pip install -r requirements.txt
```
   This will install all necessary packages, including `sentence-transformers`, `chromadb`, `ollama`, `python-dotenv`, `pymupdf4llm` (for PDF processing), `einops`, and `sentencepiece`.

3. Make sure you have Ollama installed with the required models:
```bash
ollama pull codellama:7b
```

## Usage

### Indexing Your Codebase

To index your codebase (supports Python, JS, Java, HTML, CSS, MD, TXT, PDF files by default):

```bash
python src/main.py --index /path/to/your/codebase
```

This will:
- Traverse all supported files in the directory
- Split them into meaningful chunks
- Generate embeddings using the configured sentence-transformer model
- Store them in a local ChromaDB database

### Topic-Focused Chat with Your Codebase

Once your codebase is indexed, you can start a focused chat session about a specific topic:

```bash
python src/main.py --query "How does the authentication system work?"
```

This will:
1. Start a chat session focused on your initial query (in this case, authentication)
2. Retrieve relevant code snippets for the initial query only
3. Generate a streaming response using the language model
4. Allow you to ask follow-up questions while maintaining context of the initial topic
5. Follow-up questions will reuse the initially retrieved contexts without performing additional RAG searches

This approach offers several advantages:
- Faster responses for follow-up questions (no repeated retrieval)
- Consistent context throughout the conversation
- Streaming responses for better user experience

The system maintains conversation history to provide continuity while ensuring the discussion stays focused on the original topic.

Type `exit`, `quit`, or press Ctrl+C to end the session.

All configurations (embedding models, LLM models, chunking parameters, database paths, etc.) are managed through the `src/utils/.env` file.

## System Components

- **CodeIndexer**: Handles traversing directories and chunking code files
- **EmbeddingGenerator**: Generates vector embeddings using sentence-transformers (e.g., `BAAI/bge-en-icl`)
- **ChromaDBHandler**: Manages storage and retrieval of code chunks
- **OllamaGenerator**: Handles generation of responses using local LLMs

## Extending the System

- Support additional file types by modifying the `traverse_directory` method in `indexer.py`
- Implement additional chunking strategies based on your specific needs
- Try different embedding models for better code understanding
- Customize prompt templates in the `generator.py` file
