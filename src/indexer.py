from pathlib import Path
from typing import List, Tuple, Dict, Generator
import pymupdf4llm
import re
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, Colors

class Indexer:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP

    def get_supported_file_types(self) -> List[str]:
        return ['.pdf']

    def traverse_directory(self, directory: str) -> Generator[Tuple[str, str], None, None]:
        directory_path = Path(directory)
        for file_path in directory_path.rglob('*.pdf'):
            try:
                content = self._read_pdf_file(file_path)
                if content:
                    yield str(file_path), content
            except Exception as e:
                print(f"{Colors.RED}Error reading file {file_path}: {e}{Colors.ENDC}")
    
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
                page_num = chunk['metadata']['page'] + 1  # 1-based
                parts.append(f"<!-- PAGE {page_num} -->\n{chunk['text']}")
            return '\n\n'.join(parts)
        except Exception as e:
            print(f"{Colors.RED}Error extracting text from PDF {file_path}: {e}{Colors.ENDC}")
            return ""
                
    def _chunk_markdown_content(self, file_path: str, content: str) -> List[Dict[str, str]]:
        """
        Smart chunking for markdown content that respects structural elements.
        
        Args:
            file_path: The path of the file
            content: The markdown content to chunk
            
        Returns:
            List of dictionaries containing chunks with metadata
        """
        chunks = []
        lines = content.split('\n')
        current_chunk_lines = []
        current_chunk_size = 0
        start_line = 1
        current_page = 1
        chunk_start_page = 1

        i = 0
        while i < len(lines):
            line = lines[i]

            # Track page numbers from markers inserted during PDF extraction
            if line.startswith('<!-- PAGE ') and line.endswith(' -->'):
                try:
                    current_page = int(line[10:-4])
                except ValueError:
                    pass
                i += 1
                continue

            # Check if we need to start a new chunk due to size
            if current_chunk_size > self.chunk_size and current_chunk_lines:
                # Find a good breaking point
                break_point = self._find_markdown_break_point(current_chunk_lines)
                if break_point > 0:
                    # Create chunk with content up to break point
                    chunk_content = '\n'.join(current_chunk_lines[:break_point])
                    chunks.append({
                        'file_path': file_path,
                        'start_line': start_line,
                        'end_line': start_line + break_point - 1,
                        'content': chunk_content,
                        'page_number': chunk_start_page,
                    })

                    # Calculate overlap
                    overlap_lines = self._calculate_markdown_overlap(current_chunk_lines[break_point:])
                    current_chunk_lines = overlap_lines
                    current_chunk_size = sum(len(line) + 1 for line in overlap_lines)
                    start_line = start_line + break_point - len(overlap_lines)
                    chunk_start_page = current_page
                else:
                    # Force break if no good point found
                    chunk_content = '\n'.join(current_chunk_lines)
                    chunks.append({
                        'file_path': file_path,
                        'start_line': start_line,
                        'end_line': start_line + len(current_chunk_lines) - 1,
                        'content': chunk_content,
                        'page_number': chunk_start_page,
                    })
                    current_chunk_lines = []
                    current_chunk_size = 0
                    start_line = i + 1
                    chunk_start_page = current_page
            
            # Handle different markdown elements
            if self._is_heading(line):
                # Add heading and collect its content
                section_lines, section_end = self._collect_heading_section(lines, i)
                current_chunk_lines.extend(section_lines)
                current_chunk_size += sum(len(l) + 1 for l in section_lines)
                i = section_end
                
            elif self._is_code_block_start(line):
                # Collect entire code block
                code_block_lines, code_end = self._collect_code_block(lines, i)
                current_chunk_lines.extend(code_block_lines)
                current_chunk_size += sum(len(l) + 1 for l in code_block_lines)
                i = code_end
                
            elif self._is_table_row(line):
                # Collect entire table
                table_lines, table_end = self._collect_table(lines, i)
                current_chunk_lines.extend(table_lines)
                current_chunk_size += sum(len(l) + 1 for l in table_lines)
                i = table_end
                
            elif self._is_list_item(line):
                # Collect list items at the same level
                list_lines, list_end = self._collect_list_section(lines, i)
                current_chunk_lines.extend(list_lines)
                current_chunk_size += sum(len(l) + 1 for l in list_lines)
                i = list_end
                
            else:
                # Regular line - collect paragraph
                if line.strip():  # Non-empty line
                    para_lines, para_end = self._collect_paragraph(lines, i)
                    current_chunk_lines.extend(para_lines)
                    current_chunk_size += sum(len(l) + 1 for l in para_lines)
                    i = para_end
                else:
                    # Empty line - preserve as paragraph separator
                    current_chunk_lines.append(line)
                    current_chunk_size += len(line) + 1
                    i += 1
        
        # Add the final chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunks.append({
                'file_path': file_path,
                'start_line': start_line,
                'end_line': start_line + len(current_chunk_lines) - 1,
                'content': chunk_content,
                'page_number': chunk_start_page,
            })
        
        return chunks
    
    def _is_heading(self, line: str) -> bool:
        """Check if line is a markdown heading."""
        return bool(re.match(r'^#{1,6}\s+', line.strip()))
    
    def _is_code_block_start(self, line: str) -> bool:
        """Check if line starts a code block."""
        return line.strip().startswith('```') or line.strip().startswith('~~~')
    
    def _is_table_row(self, line: str) -> bool:
        """Check if line is part of a markdown table."""
        return '|' in line.strip() and len(line.strip()) > 1
    
    def _is_list_item(self, line: str) -> bool:
        """Check if line is a list item."""
        stripped = line.strip()
        return bool(re.match(r'^[-*+]\s+', stripped) or re.match(r'^\d+\.\s+', stripped))
    
    def _collect_heading_section(self, lines: List[str], start: int) -> Tuple[List[str], int]:
        """Collect a heading and its immediate content until next heading or end."""
        section_lines = [lines[start]]  # Include the heading
        i = start + 1
        
        # Collect content until next heading of same or higher level
        heading_level = len(lines[start].split()[0])  # Count # characters
        
        while i < len(lines):
            line = lines[i]
            if self._is_heading(line):
                current_level = len(line.split()[0])
                if current_level <= heading_level:
                    break  # Found same or higher level heading
            section_lines.append(line)
            i += 1
            
            # Limit section size to prevent huge chunks
            if len(section_lines) > 50:
                break
        
        return section_lines, i
    
    def _collect_code_block(self, lines: List[str], start: int) -> Tuple[List[str], int]:
        """Collect an entire code block."""
        code_lines = [lines[start]]  # Include opening ```
        fence = lines[start].strip()[:3]  # ``` or ~~~
        i = start + 1
        
        while i < len(lines):
            code_lines.append(lines[i])
            if lines[i].strip().startswith(fence):
                i += 1
                break
            i += 1
        
        return code_lines, i
    
    def _collect_table(self, lines: List[str], start: int) -> Tuple[List[str], int]:
        """Collect an entire markdown table."""
        table_lines = []
        i = start
        
        while i < len(lines) and self._is_table_row(lines[i]):
            table_lines.append(lines[i])
            i += 1
        
        return table_lines, i
    
    def _collect_list_section(self, lines: List[str], start: int) -> Tuple[List[str], int]:
        """Collect a complete list section with nested items."""
        list_lines = []
        i = start
        base_indent = len(lines[start]) - len(lines[start].lstrip())
        
        while i < len(lines):
            line = lines[i]
            current_indent = len(line) - len(line.lstrip())
            
            # Stop if we hit a non-list line at base level or less
            if not line.strip():
                list_lines.append(line)
                i += 1
                continue
                
            if not self._is_list_item(line) and current_indent <= base_indent:
                break
                
            # Include list items and their continuation lines
            if self._is_list_item(line) or current_indent > base_indent:
                list_lines.append(line)
            else:
                break
                
            i += 1
        
        return list_lines, i
    
    def _collect_paragraph(self, lines: List[str], start: int) -> Tuple[List[str], int]:
        """Collect a complete paragraph (until empty line or special element)."""
        para_lines = []
        i = start
        
        while i < len(lines):
            line = lines[i]
            
            # Stop at empty line
            if not line.strip():
                break
                
            # Stop at special markdown elements
            if (self._is_heading(line) or self._is_code_block_start(line) or 
                self._is_table_row(line) or self._is_list_item(line)):
                break
                
            para_lines.append(line)
            i += 1
        
        return para_lines, i
    
    def _find_markdown_break_point(self, lines: List[str]) -> int:
        """Find the best point to break a chunk, preferring structural boundaries."""
        # Look for good break points from the end
        for i in range(len(lines) - 1, 0, -1):
            line = lines[i]
            prev_line = lines[i-1] if i > 0 else ""
            
            # Prefer breaking after headings
            if self._is_heading(prev_line):
                return i
                
            # Break after complete code blocks
            if prev_line.strip().startswith('```') or prev_line.strip().startswith('~~~'):
                return i
                
            # Break after empty lines (paragraph boundaries)
            if not prev_line.strip() and line.strip():
                return i
        
        # If no good break point found, break at 75% of chunk
        return int(len(lines) * 0.75)
    
    def _calculate_markdown_overlap(self, remaining_lines: List[str]) -> List[str]:
        """Calculate overlap for markdown, preserving complete elements."""
        if not remaining_lines:
            return []
            
        overlap_lines = []
        overlap_size = 0
        
        # Try to include complete elements in overlap
        for line in remaining_lines:
            if overlap_size + len(line) + 1 <= self.chunk_overlap:
                overlap_lines.append(line)
                overlap_size += len(line) + 1
            else:
                break
                
        return overlap_lines
                
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
