"""
Semantic Chunking Module

This module implements a production-grade chunking strategy for long,
structured financial PDFs. It is optimized for:

- LLM grounding (answers must be fully supported by the chunk)
- Q/A generation for fine-tuning
- High-quality retrieval (vector + hybrid search)

Key design choices:
- Sentence-aware splitting (NLP-based where available, with a robust regex
  fallback) to avoid mid-sentence breaks.
- Section-aware chunk assembly using detected headers and titles.
- Preservation of tables and bullet lists as atomic blocks.
- Token-budget-based chunk assembly with overlap for context continuity.
- Rich metadata on each chunk for downstream evaluation and debugging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .semantic_types import ChunkMetadata, PageSpan, TextUnit, infer_page_range

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class SemanticChunkingConfig:
    """
    Configuration for semantic, token-budget-aware chunking.

    The defaults are tuned for:
    - Target chunk size: ~300–600 tokens
    - Overlap: ~15%
    """

    min_tokens: int = 300
    max_tokens: int = 600
    target_tokens: int = 450
    overlap_ratio: float = 0.15  # 10–20% recommended

    # Controls for sentence splitting and block detection
    preserve_tables_as_blocks: bool = True
    preserve_bullets_as_blocks: bool = True

    # Optional: remove boilerplate lines (page numbers, legal footers, etc.)
    strip_boilerplate: bool = True


def _default_token_counter(text: str) -> int:
    """
    Very lightweight token counter that approximates LLM/BPE tokens.

    For production, we recommend passing a true tokenizer, e.g. a Hugging Face
    tokenizer or tiktoken encoder, via the `token_counter` argument.
    """
    # Heuristic: word count * 1.3 approximates BPE tokens reasonably well
    words = re.findall(r"\S+", text)
    return max(1, int(len(words) * 1.3))


def _load_spacy_sentence_segmenter() -> Optional[Callable[[str], List[Tuple[int, int]]]]:
    """
    Try to load a spaCy pipeline for robust multilingual sentence segmentation.

    Returns a function that maps text -> list of (start_char, end_char) sentence spans,
    or None if spaCy is not available.
    """
    try:
        import spacy  # type: ignore

        # Use a small English model if available; fall back to generic "en_core_web_sm"
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            # As a last resort, use the blank model with sentencizer
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")

        def segment(text: str) -> List[Tuple[int, int]]:
            doc = nlp(text)
            return [(sent.start_char, sent.end_char) for sent in doc.sents]

        return segment
    except Exception:
        return None


_SPACY_SEGMENTER = _load_spacy_sentence_segmenter()


def _regex_sentence_spans(text: str) -> List[Tuple[int, int]]:
    """
    Regex-based sentence splitter used as a fallback when spaCy is unavailable.

    It is designed to be conservative in a financial/regulatory context:
    - Avoids splitting on common abbreviations (e.g., "U.S.", "Inc.", "Ltd.")
    - Prefers splitting on punctuation followed by whitespace + capital letter.
    """
    if not text.strip():
        return []

    # A basic but robust pattern; in production you might extend the abbreviation list.
    abbreviations = {
        "mr.", "ms.", "mrs.", "dr.", "u.s.", "inc.", "ltd.", "co.", "corp.", "jan.", "feb.",
        "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.", "nov.", "dec.",
    }

    spans: List[Tuple[int, int]] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in ".!?" and i + 1 < n:
            # Look ahead for end of sentence pattern: punctuation + whitespace + capital or digit
            j = i + 1
            while j < n and text[j].isspace():
                j += 1

            # Extract potential abbreviation token
            token_start = i
            while token_start > start and text[token_start - 1].isalpha():
                token_start -= 1
            token = text[token_start : i + 1].lower()

            is_abbrev = token in abbreviations
            next_char = text[j] if j < n else ""

            if not is_abbrev and (next_char.isupper() or next_char.isdigit() or next_char == ""):
                end = j if j <= n else n
                spans.append((start, end))
                start = end
                i = end
                continue

        i += 1

    if start < n:
        spans.append((start, n))
    return spans


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    """
    Unified sentence span API that prefers spaCy when available.
    """
    if _SPACY_SEGMENTER is not None:
        return _SPACY_SEGMENTER(text)
    return _regex_sentence_spans(text)


def _is_probable_header(line: str) -> bool:
    """
    Heuristic for detecting section headers/titles in financial reports.
    """
    stripped = line.strip()
    if not stripped:
        return False

    if len(stripped) > 120:
        return False

    # Strong signals: all caps or leading numbered sections
    if stripped.isupper() and any(c.isalpha() for c in stripped):
        return True

    if re.match(r"^(item\s+\d+(\.\d+)*|[0-9]+\.)\s", stripped, flags=re.IGNORECASE):
        return True

    # Short title-like phrases
    if len(stripped.split()) <= 8 and stripped.istitle():
        return True

    return False


def _is_bullet_line(line: str) -> bool:
    return bool(re.match(r"^\s*([-*•]\s+|[0-9]+[\.\)]\s+|[a-zA-Z][\.\)]\s+)", line))


def _is_probable_table_block(block: str) -> bool:
    """
    Very lightweight table detector: looks for column delimiters and repeated
    numeric patterns. The goal is to avoid splitting tables mid-row.
    """
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False

    # Any explicit table character is a strong signal
    if any("|" in ln for ln in lines):
        return True

    # Heuristic: at least two lines with multiple numeric columns separated by 2+ spaces
    numeric_row_count = 0
    for ln in lines:
        cols = re.split(r"\s{2,}", ln.strip())
        if len(cols) >= 3 and sum(1 for c in cols if re.search(r"\d", c)) >= 2:
            numeric_row_count += 1
    return numeric_row_count >= 2


def _strip_boilerplate_lines(text: str) -> str:
    """
    Remove low-information lines such as page numbers and legal footers
    that may have survived the initial PDF cleaning step.
    """
    filtered: List[str] = []
    for ln in text.splitlines():
        raw = ln.strip()
        lower = raw.lower()

        if not raw:
            filtered.append(ln)
            continue

        # Pure page numbers or "Page X of Y"
        if re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", lower):
            continue
        if re.fullmatch(r"\d{1,4}", raw):
            # likely just a page number or table row index
            continue

        # Common legal/boilerplate snippets
        if any(
            phrase in lower
            for phrase in [
                "all rights reserved",
                "forward-looking statement",
                "safe harbor",
                "this presentation does not constitute",
                "no offer or solicitation",
            ]
        ):
            continue

        filtered.append(ln)

    return "\n".join(filtered)


def split_into_semantic_units(
    text: str,
    *,
    config: Optional[SemanticChunkingConfig] = None,
    page_spans: Optional[Sequence[PageSpan]] = None,
) -> List[TextUnit]:
    """
    Perform sentence-aware, structure-aware splitting of a cleaned document
    into semantic units.

    This function:
    - Keeps tables as atomic blocks when detected.
    - Keeps bullet lists as atomic blocks.
    - Detects probable section headers and tracks them as separate units.
    - Splits regular prose into sentences using NLP where available.

    The returned units retain absolute character offsets with respect to the
    input `text`. If `page_spans` are provided, each unit is also tagged with
    an approximate page number.
    """
    if config is None:
        config = SemanticChunkingConfig()

    if not text or not text.strip():
        return []

    working_text = _strip_boilerplate_lines(text) if config.strip_boilerplate else text

    units: List[TextUnit] = []
    offset = 0  # running character offset within original text

    # We treat double newlines as paragraph separators; this tends to preserve
    # local structure from PDFs reasonably well.
    paragraphs = re.split(r"\n{2,}", working_text)

    for para in paragraphs:
        para_len = len(para)
        if not para.strip():
            offset += para_len + 2  # account for the split delimiter (approx.)
            continue

        para_start = offset
        para_end = offset + para_len

        # Determine a representative page for the paragraph, if possible
        page: Optional[int] = None
        if page_spans:
            page_start, page_end = infer_page_range(para_start, para_end, page_spans)
            page = page_start or page_end

        # First pass: split into lines to detect headers, bullets, and tables
        lines = para.splitlines()
        joined = "\n".join(lines)

        if config.preserve_tables_as_blocks and _is_probable_table_block(joined):
            units.append(
                TextUnit(
                    kind="table_block",
                    text=joined.strip("\n"),
                    start_char=para_start,
                    end_char=para_end,
                    section_hint=None,
                    page=page,
                )
            )
            offset += para_len + 2
            continue

        if config.preserve_bullets_as_blocks and all(
            _is_bullet_line(ln) or not ln.strip() for ln in lines
        ):
            units.append(
                TextUnit(
                    kind="bullet_block",
                    text=joined.strip("\n"),
                    start_char=para_start,
                    end_char=para_end,
                    section_hint=None,
                    page=page,
                )
            )
            offset += para_len + 2
            continue

        # Check if this entire paragraph is a header
        header_candidates = [ln for ln in lines if _is_probable_header(ln)]
        if len(header_candidates) == 1 and len(lines) <= 2:
            header_text = header_candidates[0].strip()
            units.append(
                TextUnit(
                    kind="header",
                    text=header_text,
                    start_char=para_start,
                    end_char=para_end,
                    section_hint=header_text,
                    page=page,
                )
            )
            offset += para_len + 2
            continue

        # Fallback: sentence-level splitting for normal prose
        for sent_start_rel, sent_end_rel in _sentence_spans(para):
            sent_start = para_start + sent_start_rel
            sent_end = para_start + sent_end_rel
            sent_text = text[sent_start:sent_end].strip()
            if not sent_text:
                continue

            units.append(
                TextUnit(
                    kind="sentence",
                    text=sent_text,
                    start_char=sent_start,
                    end_char=sent_end,
                    section_hint=None,
                    page=page,
                )
            )

        offset += para_len + 2

    return units


def assemble_chunks_with_overlap(
    units: Sequence[TextUnit],
    *,
    config: Optional[SemanticChunkingConfig] = None,
    token_counter: Optional[TokenCounter] = None,
    page_spans: Optional[Sequence[PageSpan]] = None,
) -> List[Dict[str, Any]]:
    """
    Assemble semantic units into overlapping chunks under a token budget.

    - Chunks are constructed so that:
      - They do not break individual `TextUnit`s (sentences, bullet blocks, tables).
      - They aim for `config.target_tokens` within [min_tokens, max_tokens].
      - Overlap is enforced at the *unit* level, preserving coherent context
        rather than raw token slices.
    - Each chunk is returned as a dict with:
      - 'chunk_id', 'text'
      - 'start_char', 'end_char'
      - 'metadata' (includes section + page range)
    """
    if config is None:
        config = SemanticChunkingConfig()
    if token_counter is None:
        token_counter = _default_token_counter

    chunks: List[Dict[str, Any]] = []
    if not units:
        return chunks

    current_units: List[TextUnit] = []
    current_tokens = 0
    chunk_id = 0
    current_section: Optional[str] = None

    def flush_chunk(force: bool = False) -> None:
        nonlocal chunk_id, current_units, current_tokens, current_section
        if not current_units:
            return

        if not force and current_tokens < max(config.min_tokens, int(0.5 * config.target_tokens)):
            return

        chunk_text = "\n".join(u.text for u in current_units)
        start_char = current_units[0].start_char
        end_char = current_units[-1].end_char

        # Determine section as the last seen header before or inside the chunk
        section = current_section
        for u in current_units:
            if u.kind == "header" and u.section_hint:
                section = u.section_hint

        page_start, page_end = infer_page_range(start_char, end_char, page_spans)

        meta = ChunkMetadata(
            chunk_id=chunk_id,
            start_char=start_char,
            end_char=end_char,
            source_section=section,
            page_start=page_start,
            page_end=page_end,
            num_tokens=current_tokens,
            num_units=len(current_units),
        )

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "start_char": start_char,
                "end_char": end_char,
                "metadata": meta.to_dict(),
            }
        )
        chunk_id += 1

        # Compute overlap suffix for continuity
        overlap_tokens_target = int(current_tokens * config.overlap_ratio)
        if overlap_tokens_target <= 0:
            current_units = []
            current_tokens = 0
            return

        suffix_units: List[TextUnit] = []
        tokens_acc = 0
        # Walk backwards until we reach the overlap budget
        for u in reversed(current_units):
            u_tokens = token_counter(u.text)
            suffix_units.append(u)
            tokens_acc += u_tokens
            if tokens_acc >= overlap_tokens_target:
                break
        suffix_units.reverse()

        current_units = suffix_units
        current_tokens = sum(token_counter(u.text) for u in current_units)

    for unit in units:
        # Update section state when we encounter headers
        if unit.kind == "header" and unit.section_hint:
            current_section = unit.section_hint

        unit_tokens = token_counter(unit.text)

        # If a single unit is too large, we still include it as its own chunk.
        if unit_tokens >= config.max_tokens and not current_units:
            current_units = [unit]
            current_tokens = unit_tokens
            flush_chunk(force=True)
            current_units = []
            current_tokens = 0
            continue

        # Check if adding this unit would overflow the budget
        if current_tokens + unit_tokens > config.max_tokens and current_units:
            flush_chunk(force=False)

        current_units.append(unit)
        current_tokens += unit_tokens

    # Flush remaining units
    flush_chunk(force=True)

    return chunks


def semantic_chunk_text(
    text: str,
    *,
    config: Optional[SemanticChunkingConfig] = None,
    token_counter: Optional[TokenCounter] = None,
    page_spans: Optional[Sequence[PageSpan]] = None,
) -> List[Dict[str, Any]]:
    """
    High-level API: perform semantic chunking on a cleaned document string.

    This is the preferred entry point for building Q/A datasets and RAG indexes.
    """
    if not text or not text.strip():
        return []

    units = split_into_semantic_units(text, config=config, page_spans=page_spans)
    return assemble_chunks_with_overlap(
        units,
        config=config,
        token_counter=token_counter,
        page_spans=page_spans,
    )


def _character_chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: int = 0,
) -> List[Dict[str, Any]]:
    """
    Legacy character-based chunking.

    This is kept for backward compatibility and for quick experiments, but
    should not be used for production-quality Q/A generation or fine-tuning.
    """
    if not text or len(text.strip()) == 0:
        return []

    chunks: List[Dict[str, Any]] = []
    text_length = len(text)
    start = 0
    chunk_id = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "start_char": start,
                    "end_char": end,
                    # No rich metadata available in legacy mode
                    "metadata": {
                        "chunk_id": chunk_id,
                        "start_char": start,
                        "end_char": end,
                        "source_section": None,
                        "page_start": None,
                        "page_end": None,
                        "num_tokens": len(chunk_text),
                        "num_units": 1,
                    },
                }
            )
            chunk_id += 1

        if end >= text_length:
            break

        start = max(0, end - overlap)

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    overlap: float = 0.0,
    *,
    strategy: str = "semantic",
    token_counter: Optional[TokenCounter] = None,
    page_spans: Optional[Sequence[PageSpan]] = None,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible public chunking API.

    Parameters
    ----------
    text:
        Cleaned document text.
    chunk_size:
        - When `strategy="character"`: maximum characters per chunk.
        - When `strategy="semantic"`: approximate *target* token count
          for each chunk (within the [min_tokens, max_tokens] bounds).
    overlap:
        - When `strategy="character"`: number of overlapping characters.
        - When `strategy="semantic"`: fractional overlap ratio in [0, 1].
    strategy:
        "semantic" (default) or "character".
    token_counter:
        Optional callable that returns the token count for a text segment.
        If omitted, a lightweight heuristic counter is used.
    page_spans:
        Optional page-span mapping for accurate page_start/page_end metadata.

    Returns
    -------
    List[Dict[str, Any]]
        Each dict contains:
        - 'chunk_id'
        - 'text'
        - 'start_char'
        - 'end_char'
        - 'metadata': see `ChunkMetadata.to_dict()`
    """
    if strategy == "character":
        return _character_chunk_text(text, chunk_size=chunk_size, overlap=int(overlap))

    # Semantic mode
    min_tokens = int(0.6 * chunk_size)
    max_tokens = int(1.33 * chunk_size)
    overlap_ratio = float(overlap) if 0.0 < overlap < 1.0 else 0.15

    sem_config = SemanticChunkingConfig(
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        target_tokens=chunk_size,
        overlap_ratio=overlap_ratio,
    )

    return semantic_chunk_text(
        text,
        config=sem_config,
        token_counter=token_counter,
        page_spans=page_spans,
    )
