"""
pipeline/chunking.py

Semantic chunking pipeline — replaces Claude API-based extraction.

5 stages:
  1. Pre-clean  — strip page numbers, repeated headers/footers, pure table rows
  2. Tokenize   — paragraph-aware sentence splitting (NLTK)
  3. Embed      — sentence-transformers/all-MiniLM-L6-v2 (80 MB, CPU-friendly)
  4. Boundaries — cosine similarity drop → topic boundary detection
  5. Pack       — group sentences into ≤400-token chunks with 1-sentence overlap
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

from utils.logger import get_logger

nltk.download("punkt_tab", quiet=True)
logger = get_logger("chunking")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-document-type cosine similarity thresholds for boundary detection.
# Lower = split more aggressively.
_THRESHOLDS: dict[str, float] = {
    "equity_research": 0.45,
    "earnings_transcript": 0.40,
    "10K": 0.45,
    "10Q": 0.45,
    "MDA": 0.45,
    "news": 0.50,
    "press_release": 0.50,
    "filing": 0.45,
    "other": 0.45,
}

_TARGET_TOKENS = 180    # soft target per chunk — smaller = more focused = clearer FinBERT signal
_MIN_TOKENS = 30        # orphan chunks below this are merged forward
_MIN_SENTENCES = 2      # minimum sentences before a boundary can be placed
_EMBED_BATCH = 64       # sentence embedding batch size

# Lazy-loaded module-level singletons (used when model is not injected from app.py)
_sentence_model = None
_bert_tokenizer = None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    chunk_id: int
    text: str
    section: str
    page_hint: Optional[int]


# ---------------------------------------------------------------------------
# Stage 1: Pre-clean
# ---------------------------------------------------------------------------

def _is_table_row(line: str) -> bool:
    """True if >60% of non-space characters are numeric/table punctuation."""
    stripped = line.strip()
    if not stripped or len(stripped) < 5:
        return False
    table_chars = sum(1 for c in stripped if c in "0123456789|$%,.-() ")
    return table_chars / len(stripped) > 0.70


def _detect_repeated_lines(lines: list[str]) -> set[str]:
    """Return short lines that appear 3+ times — likely page headers/footers."""
    counts = Counter(l.strip() for l in lines if 0 < len(l.strip()) <= 60)
    return {line for line, count in counts.items() if count >= 3}


def _clean_text(raw_text: str) -> str:
    """Remove page numbers, repeated headers/footers, and pure table rows."""
    lines = raw_text.split("\n")
    repeated = _detect_repeated_lines(lines)

    kept = []
    for line in lines:
        stripped = line.strip()
        # Standalone page number
        if re.match(r"^\d{1,4}$", stripped):
            continue
        # Repeated header/footer
        if stripped in repeated:
            continue
        # Pure table row (no narrative content)
        if _is_table_row(stripped):
            continue
        kept.append(line)

    # Collapse runs of 3+ blank lines to 2
    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Stage 2: Sentence tokenization
# ---------------------------------------------------------------------------

def _tokenize_sentences(text: str) -> list[str]:
    """
    Split on paragraph breaks first, then sentence-tokenize within each paragraph.
    Paragraph breaks are stronger boundaries than sentence boundaries.
    Filters out sentences under 15 characters (orphan words, page artifacts).
    """
    paragraphs = re.split(r"\n{2,}", text)
    sentences: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        for sent in sent_tokenize(para):
            sent = sent.strip()
            if len(sent) >= 15:
                sentences.append(sent)
    return sentences


# ---------------------------------------------------------------------------
# Stage 3: Embedding
# ---------------------------------------------------------------------------

def _get_sentence_model(model):
    """Return the injected model, or load the module-level singleton."""
    global _sentence_model
    if model is not None:
        return model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers/all-MiniLM-L6-v2 ...")
        _sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _sentence_model


def _get_bert_tokenizer():
    global _bert_tokenizer
    if _bert_tokenizer is None:
        _bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return _bert_tokenizer


def _count_tokens(text: str) -> int:
    tok = _get_bert_tokenizer()
    return len(tok.encode(text, add_special_tokens=True))


def _embed(sentences: list[str], model) -> np.ndarray:
    """Return (N, D) float32 embeddings, L2-normalised."""
    embeddings = model.encode(
        sentences,
        batch_size=_EMBED_BATCH,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 4: Boundary detection
# ---------------------------------------------------------------------------

def _detect_boundaries(embeddings: np.ndarray, threshold: float) -> list[int]:
    """
    Return indices where a new chunk should start (boundary before sentence[i]).
    Cosine similarity of normalised vectors = dot product.
    """
    boundaries = [0]  # first sentence always starts a chunk
    n = len(embeddings)

    for i in range(1, n):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]))
        sentences_since_boundary = i - boundaries[-1]
        if sim < threshold and sentences_since_boundary >= _MIN_SENTENCES:
            boundaries.append(i)

    return boundaries


# ---------------------------------------------------------------------------
# Stage 5: Section labelling
# ---------------------------------------------------------------------------

_HEADING_PATTERNS = [
    re.compile(r"^[A-Z][A-Z\s\-\/]{4,79}$"),          # ALL CAPS line
    re.compile(r"^\d+[\.\)]\s+[A-Z]"),                  # "3. Risk Factors"
    re.compile(r"^[A-Z][^.!?]{3,78}:$"),                # "Forward-Looking Statements:"
]


def _detect_section_label(sentences: list[str]) -> str:
    """Scan sentences for a heading; fall back to first-words summary."""
    for sent in sentences:
        s = sent.strip()
        for pattern in _HEADING_PATTERNS:
            if pattern.match(s) and len(s) <= 80:
                # Title-case the label for consistency
                return s.title() if s.isupper() else s.rstrip(":")
    # Fallback: first 6 words of the chunk
    words = " ".join(sentences).split()
    return " ".join(words[:6]) + "..." if len(words) > 6 else " ".join(words)


# ---------------------------------------------------------------------------
# Stage 5: Chunk packing
# ---------------------------------------------------------------------------

def _pack_chunks(
    sentences: list[str],
    boundaries: list[int],
    estimated_pages: int,
    filename: str,
) -> list[ChunkResult]:
    """
    Pack sentence groups into ChunkResult objects respecting _TARGET_TOKENS.
    Applies 1-sentence overlap between chunks for context continuity.
    Merges orphan chunks (< _MIN_TOKENS) forward.
    """
    # Build sentence groups from boundaries
    groups: list[list[str]] = []
    boundary_set = set(boundaries)
    current: list[str] = []

    for i, sent in enumerate(sentences):
        if i in boundary_set and current:
            groups.append(current)
            # 1-sentence overlap: carry last sentence into next group
            current = [current[-1], sent]
        else:
            current.append(sent)
    if current:
        groups.append(current)

    # Within each group, enforce _TARGET_TOKENS by further splitting if needed
    final_sentences_groups: list[list[str]] = []
    for group in groups:
        current_sents: list[str] = []
        current_tokens = 0
        for sent in group:
            sent_tokens = _count_tokens(sent)
            if current_tokens + sent_tokens > _TARGET_TOKENS and current_sents:
                final_sentences_groups.append(current_sents)
                current_sents = [sent]
                current_tokens = sent_tokens
            else:
                current_sents.append(sent)
                current_tokens += sent_tokens
        if current_sents:
            final_sentences_groups.append(current_sents)

    # Merge orphan groups (< _MIN_TOKENS) into the next group
    merged: list[list[str]] = []
    for group in final_sentences_groups:
        token_count = _count_tokens(" ".join(group))
        if token_count < _MIN_TOKENS and merged:
            merged[-1].extend(group)
        else:
            merged.append(group)

    # Build ChunkResult objects
    chunks: list[ChunkResult] = []
    total = max(len(merged), 1)
    for idx, group in enumerate(merged):
        text = " ".join(group).strip()
        if not text:
            continue
        page_hint = max(1, round((idx / total) * estimated_pages))
        chunks.append(ChunkResult(
            chunk_id=idx + 1,
            text=text,
            section=_detect_section_label(group),
            page_hint=page_hint,
        ))

    logger.debug(f"Packed {len(groups)} boundary-groups → {len(chunks)} final chunks")
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_chunks(
    filename: str,
    raw_text: str,
    document_type: str,
    model=None,
) -> list[ChunkResult]:
    """
    Semantic chunking pipeline. Zero API calls.

    Args:
        filename:       For logging only.
        raw_text:       Full extracted document text.
        document_type:  Used to select the similarity threshold.
        model:          Optional pre-loaded SentenceTransformer (injected from app.py).
                        Falls back to module-level lazy singleton if None.

    Returns:
        list[ChunkResult] — ready to pass to token_enforcement.enforce_token_limit()
    """
    if not raw_text or not raw_text.strip():
        logger.warning(f"{filename}: empty raw_text, returning no chunks")
        return []

    threshold = _THRESHOLDS.get(document_type, 0.45)
    estimated_pages = max(1, len(raw_text) // 3000)

    logger.info(f"{filename}: cleaning text ({len(raw_text):,} chars)")
    clean = _clean_text(raw_text)

    logger.info(f"{filename}: tokenizing sentences")
    sentences = _tokenize_sentences(clean)
    if not sentences:
        logger.warning(f"{filename}: no sentences after cleaning")
        return []

    logger.info(f"{filename}: embedding {len(sentences)} sentences")
    sent_model = _get_sentence_model(model)
    embeddings = _embed(sentences, sent_model)

    logger.info(f"{filename}: detecting boundaries (threshold={threshold})")
    boundaries = _detect_boundaries(embeddings, threshold)

    logger.info(f"{filename}: packing chunks ({len(boundaries)} boundaries detected)")
    chunks = _pack_chunks(sentences, boundaries, estimated_pages, filename)

    logger.info(f"{filename}: produced {len(chunks)} chunks from {len(sentences)} sentences")
    return chunks
