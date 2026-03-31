from dataclasses import dataclass
from typing import Callable, Optional
from transformers import pipeline as hf_pipeline
from utils.logger import get_logger

logger = get_logger("finbert")


@dataclass
class SentimentResult:
    chunk_id: int
    positive: float
    negative: float
    neutral: float
    label: str
    confidence: float


_finbert = None


def get_finbert():
    global _finbert
    if _finbert is None:
        logger.info("Loading FinBERT model (ProsusAI/finbert)...")
        _finbert = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,  # return all 3 label scores
            truncation=True,
            max_length=512,
        )
    return _finbert


def run_finbert(
    chunks: list[tuple],  # list of (ChunkResult, token_count)
    batch_size: int = 16,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[SentimentResult]:
    """
    Run FinBERT on all chunks. Returns SentimentResult per chunk.
    chunks: list of (ChunkResult, token_count) from token_enforcement
    """
    model = get_finbert()
    results = []

    texts = [c.text for c, _ in chunks]
    chunk_ids = [c.chunk_id for c, _ in chunks]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_ids = chunk_ids[i:i + batch_size]

        raw_outputs = model(batch)  # list of list of {label, score}

        for chunk_id, output in zip(batch_ids, raw_outputs):
            scores = {item["label"].lower(): item["score"] for item in output}
            positive = scores.get("positive", 0.0)
            negative = scores.get("negative", 0.0)
            neutral = scores.get("neutral", 0.0)
            label = max(scores, key=scores.get)
            confidence = scores[label]

            results.append(SentimentResult(
                chunk_id=chunk_id,
                positive=positive,
                negative=negative,
                neutral=neutral,
                label=label,
                confidence=confidence,
            ))

        if progress_callback:
            progress_callback(min(i + batch_size, len(texts)), len(texts))

    return results
