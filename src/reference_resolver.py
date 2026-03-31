import re
from typing import List, Dict, Any, Set

from src.config import Colors


# Patterns for Dutch and English cross-references found in legal/HR documents
_REF_PATTERNS = [
    # Dutch: artikel 5, artikel 5.1, art. 3
    re.compile(r'\b(?:artikel|art\.?)\s+(\d+(?:\.\d+)*)', re.IGNORECASE),
    # Dutch: hoofdstuk 3, bijlage A/1
    re.compile(r'\b(?:hoofdstuk|bijlage|paragraaf|lid|punt)\s+([A-Za-z0-9]+(?:\.\d+)*)', re.IGNORECASE),
    # English: article 5, section 3.2, chapter 2, appendix A
    re.compile(r'\b(?:article|section|chapter|appendix|clause|annex)\s+([A-Za-z0-9]+(?:\.\d+)*)', re.IGNORECASE),
]


def extract_references(text: str) -> List[str]:
    """Extract cross-reference phrases from a chunk of text.

    Returns a list of search phrases like "artikel 5", "bijlage A".
    """
    refs: list[str] = []
    seen: Set[str] = set()
    for pattern in _REF_PATTERNS:
        for match in pattern.finditer(text):
            phrase = match.group(0).strip()
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                refs.append(phrase)
    return refs


def resolve_references(
    contexts: List[Dict[str, Any]],
    embedding_generator,
    db_handler,
    n_extra: int = 3,
    similarity_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """Scan retrieved contexts for cross-references and fetch the referenced chunks.

    Args:
        contexts: The already-retrieved context dicts (must have 'content' key).
        embedding_generator: EmbeddingGenerator instance.
        db_handler: ChromaDBHandler instance.
        n_extra: Max extra chunks to fetch per reference phrase.
        similarity_threshold: Min similarity for referenced chunks.

    Returns:
        The original contexts list, potentially extended with referenced chunks.
    """
    # Collect all references from retrieved chunks
    all_refs: list[str] = []
    seen_refs: Set[str] = set()
    for ctx in contexts:
        for ref in extract_references(ctx['content']):
            key = ref.lower()
            if key not in seen_refs:
                seen_refs.add(key)
                all_refs.append(ref)

    if not all_refs:
        return contexts

    print(f"{Colors.BLUE}Found cross-references: {', '.join(all_refs)}{Colors.ENDC}")

    # Track existing chunk content to avoid duplicates (use first 200 chars as key)
    existing_keys: Set[str] = {ctx['content'][:200] for ctx in contexts}

    added = 0
    for ref in all_refs:
        try:
            ref_embedding = embedding_generator.generate_query_embedding(ref)
            results = db_handler.query(
                ref_embedding,
                n_results=n_extra,
                similarity_threshold=similarity_threshold,
            )

            if not results or not results.get('documents') or not results['documents'][0]:
                continue

            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0],
            ):
                # Skip if we already have this chunk
                if doc[:200] in existing_keys:
                    continue

                similarity = db_handler._convert_distance_to_similarity(distance)
                contexts.append({
                    'content': doc,
                    'file_path': metadata['file_path'],
                    'start_line': metadata['start_line'],
                    'end_line': metadata['end_line'],
                    'page_number': metadata.get('page_number', 1),
                    'similarity': round(similarity, 4),
                })
                existing_keys.add(doc[:200])
                added += 1

        except Exception as e:
            print(f"{Colors.YELLOW}Could not resolve reference '{ref}': {e}{Colors.ENDC}")

    if added:
        print(f"{Colors.GREEN}Added {added} chunks from cross-references{Colors.ENDC}")

    return contexts
