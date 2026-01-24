"""
Ingestion Package
Modules for PDF loading, chunking, Q/A generation, and dataset writing.
"""

from .pdf_loader import load_pdf, clean_text
from .chunker import chunk_text
from .qa_generator import generate_qa_pairs
from .dataset_writer import (
    save_to_jsonl,
    split_dataset,
    validate_qa_pair,
    filter_valid_pairs
)

__all__ = [
    'load_pdf',
    'clean_text',
    'chunk_text',
    'generate_qa_pairs',
    'save_to_jsonl',
    'split_dataset',
    'validate_qa_pair',
    'filter_valid_pairs'
]
