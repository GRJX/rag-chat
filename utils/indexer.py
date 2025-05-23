from pathlib import Path
from typing import List, Tuple, Dict, Generator, Optional, Union, Set
import pymupdf4llm
import pymupdf
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS, Colors

class Indexer:
    def __init__(self):
        """
        Initialize the code indexer using environment configuration.
        """
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self._supported_extensions = SUPPORTED_EXTENSIONS
        
    def get_supported_file_types(self) -> List[str]:
        """
        Get list of supported file extensions
        
        Returns:
            List of supported file extensions with dots (e.g., ['.py', '.pdf'])
        """
        return list(self._supported_extensions.keys())
        
    def traverse_directory(self, directory: str, file_extensions: Union[str, List[str]] = None, auto_detect: bool = False) -> Generator[Tuple[str, str], None, None]:
        """
        Traverse a directory and yield file paths and their content.
        
        Args:
            directory: The directory to traverse
            file_extensions: File extension(s) to filter by. Can be a string, list, or None for auto-detection
            auto_detect: If True, automatically detect and process all supported file types
            
        Yields:
            Tuple of (file_path, file_content)
        """
        directory_path = Path(directory)
        
        # Auto-detect mode - use all supported extensions
        if auto_detect or file_extensions is None:
            file_extensions = self.get_supported_file_types()
        
        # Convert single extension to list for unified handling
        if isinstance(file_extensions, str):
            file_extensions = [file_extensions]
            
        for ext in file_extensions:
            for file_path in directory_path.rglob(f'*{ext}'):
                try:
                    content = self._read_file_content(file_path)
                    if content:
                        yield str(file_path), content
                except Exception as e:
                    print(f"{Colors.RED}Error reading file {file_path}: {e}{Colors.ENDC}")
    
    def _read_file_content(self, file_path: Path) -> str:
        """
        Read content from a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            String content of the file
        """
        file_extension = file_path.suffix.lower()
        
        # Handle different file types
        try:
            if file_extension == '.pdf':
                return self._read_pdf_file(file_path) # Method name kept, implementation changes
            elif file_extension in ['.txt', '.md', '.py', '.js', '.java', '.html', '.css']:
                return self._read_text_file(file_path)
            else:
                print(f"{Colors.RED}Unsupported file type: {file_extension}. Attempting to read as text.{Colors.ENDC}")
                return self._read_text_file(file_path)
        except UnicodeDecodeError:
            print(f"{Colors.RED}File {file_path} appears to be binary. Skipping.{Colors.ENDC}")
            return ""
        except Exception as e:
            print(f"{Colors.RED}Error reading file {file_path}: {e}{Colors.ENDC}")
            return ""
            
    def _read_text_file(self, file_path: Path) -> str:
        """Read a text-based file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception:
                raise
    
    def _read_pdf_file(self, file_path: Path) -> str:
        """
        Extract text content from a PDF file and convert it to Markdown
        using pymupdf4llm.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF in Markdown format
        """
        try:
            # Convert PDF to Markdown using PyMuPDF4LLM
            md_text = pymupdf4llm.to_markdown(str(file_path), table_strategy="lines")
            return md_text
        except Exception as e:
            print(f"{Colors.RED}Error extracting text from PDF {file_path} using PyMuPDF4LLM: {e}{Colors.ENDC}")
            return ""
                
    def chunk_data(self, file_path: str, content: str) -> List[Dict[str, str]]:
        """
        Split content into overlapping chunks using smart chunking for markdown files.
        
        Args:
            file_path: The path of the file
            content: The content to chunk
            
        Returns:
            List of dictionaries containing chunks with metadata
        """
        chunks = []
        
        # Split into lines to preserve context
        lines = content.split('\n')
        current_chunk = []
        current_chunk_size = 0
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            line_with_newline = line + '\n'
            line_size = len(line_with_newline)
            
            # If adding this line would exceed chunk size and we already have content,
            # save the current chunk and start a new one with overlap
            if current_chunk_size + line_size > self.chunk_size and current_chunk:
                chunk_text = ''.join(current_chunk)
                chunks.append({
                    'file_path': file_path,
                    'start_line': start_line,
                    'end_line': i - 1,
                    'content': chunk_text,
                })
                
                # Calculate overlap
                overlap_lines = []
                overlap_size = 0
                for line_idx in range(len(current_chunk) - 1, -1, -1):
                    if overlap_size + len(current_chunk[line_idx]) <= self.chunk_overlap:
                        overlap_lines.insert(0, current_chunk[line_idx])
                        overlap_size += len(current_chunk[line_idx])
                    else:
                        break
                
                # Start new chunk with overlapping lines
                current_chunk = overlap_lines
                current_chunk_size = overlap_size
                start_line = i - len(overlap_lines)
            
            current_chunk.append(line_with_newline)
            current_chunk_size += line_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ''.join(current_chunk)
            chunks.append({
                'file_path': file_path,
                'start_line': start_line,
                'end_line': len(lines),
                'content': chunk_text,
            })
            
        return chunks
