"""
Text Chunking Module
Splits documents into fixed-size chunks for processing.
"""

from typing import List, Dict


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 0
) -> List[Dict[str, any]]:
    """
    Split text into chunks of specified size.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of dictionaries, each containing:
        - 'chunk_id': Unique identifier for the chunk
        - 'text': The chunk text
        - 'start_char': Starting character position in original text
        - 'end_char': Ending character position in original text
    """
    if not text or len(text.strip()) == 0:
        return []
    
    chunks = []
    text_length = len(text)
    start = 0
    chunk_id = 0
    
    while start < text_length:
        # Calculate end position
        end = min(start + chunk_size, text_length)
        
        # Extract chunk
        chunk_text = text[start:end]
        
        # Only add non-empty chunks
        if chunk_text.strip():
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'start_char': start,
                'end_char': end
            })
            chunk_id += 1
        
        # Move to next chunk with overlap
        if end >= text_length:
            break
        
        start = end - overlap
    
    return chunks
