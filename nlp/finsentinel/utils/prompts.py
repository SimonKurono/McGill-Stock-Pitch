# CHUNKING_PROMPT removed — chunking is now handled locally by the semantic pipeline
# in pipeline/chunking.py (sentence-transformers + cosine similarity boundary detection).

CLASSIFICATION_PROMPT = """You are a financial document analyst. Given the beginning of a document,
classify it and determine if it is suitable for financial sentiment analysis.

Suitable documents: equity research reports, 10-K, 10-Q, MD&A sections, earnings call transcripts,
financial news articles, analyst commentary, company press releases with substance.

NOT suitable: pure legal disclaimers (>80% legal boilerplate), index/table of contents only,
regulatory forms with no narrative text, cover pages, glossaries, pure numerical tables.

Flag sentiment_bias_warning if the document appears to be purely promotional (e.g. investor
relations marketing brochure) with no critical analysis.

Respond ONLY with valid JSON — no explanation, no markdown, no code fences. Raw JSON only.

Schema:
{{
  "document_type": "equity_research|10K|10Q|MDA|earnings_transcript|news|press_release|filing|other",
  "suitable_for_sentiment": true or false,
  "rejection_reason": null or "explanation string",
  "sentiment_bias_warning": null or "bullish_biased" or "bearish_biased",
  "confidence": 0.0 to 1.0
}}

Document filename: {filename}
Document beginning (first 3000 characters):
---
{text_preview}
---
"""

