import nltk
from transformers import AutoTokenizer
from pipeline.chunking import ChunkResult
from utils.logger import get_logger

nltk.download("punkt_tab", quiet=True)
logger = get_logger("token_enforcement")

_tokenizer = None
MAX_TOKENS = 512


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        logger.info("Loading BERT tokenizer for token counting...")
        _tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return _tokenizer


def count_tokens(text: str) -> int:
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=True))


def split_into_sentences(text: str) -> list[str]:
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)


def split_chunk(chunk: ChunkResult, base_id: int) -> list[ChunkResult]:
    """Split an oversized chunk at sentence boundaries."""
    sentences = split_into_sentences(chunk.text)
    tok = get_tokenizer()

    sub_chunks = []
    current_sentences: list[str] = []
    current_token_count = 0
    sub_id = 0

    for sentence in sentences:
        sentence_tokens = len(tok.encode(sentence, add_special_tokens=False))
        if current_token_count + sentence_tokens + 2 > MAX_TOKENS and current_sentences:
            sub_chunks.append(ChunkResult(
                chunk_id=base_id + sub_id,
                text=" ".join(current_sentences),
                section=chunk.section,
                page_hint=chunk.page_hint,
            ))
            sub_id += 1
            current_sentences = [sentence]
            current_token_count = sentence_tokens
        else:
            current_sentences.append(sentence)
            current_token_count += sentence_tokens

    if current_sentences:
        sub_chunks.append(ChunkResult(
            chunk_id=base_id + sub_id,
            text=" ".join(current_sentences),
            section=chunk.section,
            page_hint=chunk.page_hint,
        ))

    return sub_chunks


def enforce_token_limit(chunks: list[ChunkResult]) -> list[tuple[ChunkResult, int]]:
    """
    Returns list of (ChunkResult, token_count) tuples.
    Splits any chunk exceeding MAX_TOKENS.
    """
    result = []
    offset = 0

    for chunk in chunks:
        token_count = count_tokens(chunk.text)
        if token_count <= MAX_TOKENS:
            result.append((chunk, token_count))
        else:
            logger.debug(f"Chunk {chunk.chunk_id} has {token_count} tokens — splitting")
            sub_chunks = split_chunk(chunk, base_id=chunk.chunk_id * 100 + offset)
            offset += len(sub_chunks)
            for sc in sub_chunks:
                sc_tokens = count_tokens(sc.text)
                result.append((sc, sc_tokens))

    return result
