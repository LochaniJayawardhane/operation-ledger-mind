"""
Lightweight semantic data structures used by the chunker.

These are kept in a small dedicated module so that the chunking logic
can evolve without creating circular imports inside `src.ingestion`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TextUnit:
    """
    Smallest atomic text unit that we assemble chunks from.

    A unit can be:
    - a single sentence
    - a bullet list block
    - a table block
    - a section header/title
    """

    kind: str  # "sentence" | "bullet_block" | "table_block" | "header"
    text: str
    start_char: int
    end_char: int
    section_hint: Optional[str] = None  # header text, if this unit is a header
    page: Optional[int] = None          # optional page index (1-based)


@dataclass(frozen=True)
class PageSpan:
    """
    Mapping from character span in the concatenated document text to a page
    number. This is optional and only needed if the upstream PDF loader
    exposes page-level offsets.
    """

    start_char: int
    end_char: int
    page: int  # 1-based page index


@dataclass(frozen=True)
class ChunkMetadata:
    """
    Metadata attached to each semantic chunk.
    """

    chunk_id: int
    start_char: int
    end_char: int
    source_section: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    num_tokens: int
    num_units: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "source_section": self.source_section,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "num_tokens": self.num_tokens,
            "num_units": self.num_units,
        }


def infer_page_range(
    start_char: int,
    end_char: int,
    page_spans: Optional[Sequence[PageSpan]],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Given a character span and a list of known page spans, infer the (start, end)
    page indices that cover this span.
    """
    if not page_spans:
        return None, None

    pages = [
        span.page
        for span in page_spans
        if not (end_char < span.start_char or start_char > span.end_char)
    ]
    if not pages:
        return None, None
    return min(pages), max(pages)

