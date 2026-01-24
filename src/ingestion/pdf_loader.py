"""
PDF Loading and Text Cleaning Module
Handles PDF extraction and text cleaning operations.
"""

from pathlib import Path
from typing import Optional
import re


def load_pdf(pdf_path: str | Path) -> str:
    """
    Load and extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content as a string
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If PDF extraction fails
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Try multiple PDF libraries in order of preference
    text = None
    
    # Try PyMuPDF (fitz) first - best for text extraction
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        text = "\n".join(text_parts)
        doc.close()
    except ImportError:
        pass
    except Exception as e:
        # If fitz fails, try other libraries
        pass
    
    # Fallback to pdfplumber
    if text is None or len(text.strip()) < 100:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                text = "\n".join(text_parts)
        except ImportError:
            pass
        except Exception as e:
            pass
    
    # Fallback to PyPDF2
    if text is None or len(text.strip()) < 100:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text())
            text = "\n".join(text_parts)
        except ImportError:
            pass
        except Exception as e:
            pass
    
    if text is None or len(text.strip()) < 100:
        raise ValueError(
            f"Failed to extract text from PDF. "
            f"Please install one of: PyMuPDF (fitz), pdfplumber, or PyPDF2"
        )
    
    return text


def clean_text(
    text: str,
    remove_headers: bool = True,
    remove_footers: bool = True,
    normalize_whitespace: bool = True
) -> str:
    """
    Clean extracted text by removing headers, footers, and normalizing whitespace.
    
    Args:
        text: Raw text to clean
        remove_headers: Whether to attempt header removal
        remove_footers: Whether to attempt footer removal
        normalize_whitespace: Whether to normalize whitespace
        
    Returns:
        Cleaned text
    """
    cleaned = text
    
    # Normalize whitespace first
    if normalize_whitespace:
        # Replace multiple spaces with single space
        cleaned = re.sub(r' +', ' ', cleaned)
        # Replace multiple newlines with double newline (paragraph break)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        # Replace tabs with spaces
        cleaned = cleaned.replace('\t', ' ')
        # Strip leading/trailing whitespace from each line
        lines = [line.strip() for line in cleaned.split('\n')]
        cleaned = '\n'.join(lines)
    
    # Remove headers/footers (common patterns in annual reports)
    if remove_headers or remove_footers:
        lines = cleaned.split('\n')
        filtered_lines = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Skip common header patterns
            if remove_headers:
                if any(pattern in line_lower for pattern in [
                    'annual report', 'page', 'table of contents',
                    'confidential', 'proprietary'
                ]) and len(line.strip()) < 50:
                    # Skip short header-like lines
                    continue
            
            # Skip common footer patterns
            if remove_footers:
                if any(pattern in line_lower for pattern in [
                    'page', '©', 'copyright', 'all rights reserved'
                ]) and len(line.strip()) < 50:
                    # Skip short footer-like lines
                    continue
            
            filtered_lines.append(line)
        
        cleaned = '\n'.join(filtered_lines)
    
    # Final cleanup
    if normalize_whitespace:
        # Remove excessive blank lines
        cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)
        cleaned = cleaned.strip()
    
    return cleaned
