from pathlib import Path
from typing import List, Tuple, Dict, Generator
import pymupdf4llm
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE, Colors

class Indexer:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP

        # Stage 1: Split by markdown headers, keeping headers in content
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=False,
        )

        # Stage 2: Further split large sections at natural boundaries
        # Priority order: paragraphs > lines > sentences > clauses > words
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            keep_separator=True,
        )

    def get_supported_file_types(self) -> List[str]:
        return ['.pdf', '.md']

    def traverse_directory(self, directory: str) -> Generator[Tuple[str, str], None, None]:
        directory_path = Path(directory)
        supported_extensions = self.get_supported_file_types()
        
        # Collect all files first to provide accurate progress information
        for file_path in directory_path.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    ext = file_path.suffix.lower()
                    if ext == '.pdf':
                        content = self._read_pdf_file(file_path)
                    elif ext == '.md':
                        content = self._read_text_file(file_path)
                    else:
                        continue
                        
                    if content:
                        yield str(file_path), content
                except Exception as e:
                    print(f"{Colors.RED}Error reading file {file_path}: {e}{Colors.ENDC}")
    
    def _read_text_file(self, file_path: Path) -> str:
        """
        Read content from a text-based file (e.g., Markdown).
        
        Args:
            file_path: Path to the file
            
        Returns:
            The file content as string
        """
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"{Colors.RED}Error reading text file {file_path}: {e}{Colors.ENDC}")
            return ""

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
            page_chunks = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
            parts = []
            for chunk in page_chunks:
                page_num = chunk['metadata'].get('page_number', chunk['metadata'].get('page', 0) + 1)
                parts.append(f"<!-- PAGE {page_num} -->\n{chunk['text']}")
            return '\n\n'.join(parts)
        except Exception as e:
            print(f"{Colors.RED}Error extracting text from PDF {file_path}: {e}{Colors.ENDC}")
            return ""
                
    def _build_page_map(self, content: str) -> List[Tuple[int, int]]:
        """Build a mapping of character offsets in cleaned content to page numbers."""
        page_map = []
        clean_offset = 0
        current_page = 1

        for line in content.split('\n'):
            match = re.match(r'^<!-- PAGE (\d+) -->$', line.strip())
            if match:
                current_page = int(match.group(1))
            else:
                page_map.append((clean_offset, current_page))
                clean_offset += len(line) + 1  # +1 for newline

        return page_map

    def _get_page_for_position(self, position: int, page_map: List[Tuple[int, int]]) -> int:
        """Find the page number for a given character position."""
        page = 1
        for offset, page_num in page_map:
            if offset <= position:
                page = page_num
            else:
                break
        return page

    def _chunk_markdown_content(self, file_path: str, content: str) -> List[Dict[str, str]]:
        """
        Smart chunking for markdown content using langchain text splitters.

        Two-stage approach:
        1. MarkdownHeaderTextSplitter splits by headers, preserving header context
        2. RecursiveCharacterTextSplitter further splits large sections at
           paragraph, sentence, and word boundaries (in that priority order)
        """
        # Build page map before stripping markers
        page_map = self._build_page_map(content)
        clean_content = re.sub(r'<!-- PAGE \d+ -->\n?', '', content)

        # Stage 1: Split by markdown headers
        header_docs = self._header_splitter.split_text(clean_content)

        # Stage 1.5: Merge small sections into the next section so title pages
        # and short header-only fragments don't become standalone chunks
        merged_docs = []
        carry = ""
        for doc in header_docs:
            text = doc.page_content
            if carry:
                text = carry + "\n\n" + text
                carry = ""
            if len(text.strip()) < MIN_CHUNK_SIZE:
                carry = text
            else:
                doc.page_content = text
                merged_docs.append(doc)
        # Attach any trailing small section to the last doc
        if carry and merged_docs:
            merged_docs[-1].page_content += "\n\n" + carry
        elif carry:
            from langchain_core.documents import Document
            merged_docs.append(Document(page_content=carry))

        # Stage 2: Further split large sections at natural boundaries
        final_docs = self._text_splitter.split_documents(merged_docs)

        # Convert to expected chunk format with metadata
        chunks = []
        search_start = 0
        for doc in final_docs:
            chunk_text = doc.page_content

            # Find position in clean content for line number / page tracking
            pos = clean_content.find(chunk_text[:min(200, len(chunk_text))], search_start)
            if pos == -1:
                pos = clean_content.find(chunk_text[:min(200, len(chunk_text))])
            if pos != -1:
                search_start = pos + 1

            # Compute approximate line numbers
            text_before = clean_content[:max(pos, 0)]
            start_line = text_before.count('\n') + 1
            end_line = start_line + chunk_text.count('\n')

            # Determine page number from PDF page markers
            page_num = self._get_page_for_position(max(pos, 0), page_map) if page_map else 1

            chunks.append({
                'file_path': file_path,
                'start_line': start_line,
                'end_line': end_line,
                'content': chunk_text,
                'page_number': page_num,
            })

        return chunks

    def chunk_data(self, file_path: str, content: str) -> List[Dict[str, str]]:
        """
        Split content into overlapping chunks using smart chunking for markdown content.

        Args:
            file_path: The path of the file
            content: The content to chunk

        Returns:
            List of dictionaries containing chunks with metadata
        """
        return self._chunk_markdown_content(file_path, content)
