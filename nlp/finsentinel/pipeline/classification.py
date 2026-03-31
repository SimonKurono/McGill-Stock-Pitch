from dataclasses import dataclass
from typing import Optional
from utils.prompts import CLASSIFICATION_PROMPT
from utils.claude_client import call_claude_json
from utils.logger import get_logger

logger = get_logger("classification")


@dataclass
class ClassificationResult:
    document_type: str
    suitable_for_sentiment: bool
    rejection_reason: Optional[str]
    sentiment_bias_warning: Optional[str]
    confidence: float


def classify_document(filename: str, raw_text: str) -> ClassificationResult:
    text_preview = raw_text[:3000]
    prompt = CLASSIFICATION_PROMPT.format(filename=filename, text_preview=text_preview)

    logger.info(f"Classifying: {filename}")
    result = call_claude_json(prompt)

    return ClassificationResult(
        document_type=result.get("document_type", "other"),
        suitable_for_sentiment=result.get("suitable_for_sentiment", False),
        rejection_reason=result.get("rejection_reason"),
        sentiment_bias_warning=result.get("sentiment_bias_warning"),
        confidence=result.get("confidence", 0.5),
    )
