"""
pipeline/data_sources.py

Fetches live financial news from Yahoo Finance (yfinance) and News API (newsapi.org).
Returns dicts with the same schema as ingest_file():
    {filename, file_type, raw_text, char_count, is_scanned_pdf: False}

Items with very little content are included but prefixed with _HEADLINE_ONLY_PREFIX so
the classification agent downstream rejects them with a visible reason in the UI.
"""
from __future__ import annotations

import os
import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from utils.logger import get_logger
import newsapi

logger = get_logger("data_sources")

# Items whose assembled text is below this length are flagged as headline-only.
_MIN_CHAR_THRESHOLD = 150
_HEADLINE_ONLY_PREFIX = "[headline-only] "


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assemble_yfinance_text(item: dict) -> str:
    """Build the richest available text from a yfinance news item."""
    parts: list[str] = []

    title = (item.get("title") or "").strip()
    publisher = (item.get("publisher") or "").strip()
    pub_ts = item.get("providerPublishTime")

    if pub_ts:
        try:
            pub_date = datetime.datetime.utcfromtimestamp(pub_ts).strftime("%Y-%m-%d")
        except (OSError, ValueError, OverflowError):
            pub_date = ""
    else:
        pub_date = ""

    header_parts = [x for x in [pub_date, publisher, title] if x]
    if header_parts:
        parts.append(" | ".join(header_parts))

    # Try richer body from the 'content' dict (newer yfinance API)
    has_body = False
    content = item.get("content")
    if isinstance(content, dict):
        body = (content.get("body") or "").strip()
        summary = (content.get("summary") or "").strip()
        if body:
            parts.append(body)
            has_body = True
        elif summary:
            parts.append(summary)
            has_body = True

    # Fallback: top-level description / summary (only if no richer body found above)
    if not has_body:
        description = (item.get("description") or item.get("summary") or "").strip()
        if description:
            parts.append(description)

    return "\n\n".join(parts)


def _make_doc(filename: str, raw_text: str) -> dict:
    return {
        "filename": filename,
        "file_type": "live_news",
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "is_scanned_pdf": False,
    }


def _safe_filename(text: str, max_len: int = 60) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in text)[:max_len]


def _build_fetched_doc(raw_text: str, title: str, i: int, source_prefix: str) -> dict:
    """Shared filename construction and thin-content detection for all sources."""
    safe_title = _safe_filename(title)
    is_thin = len(raw_text) < _MIN_CHAR_THRESHOLD
    prefix = _HEADLINE_ONLY_PREFIX if is_thin else ""
    filename = f"{prefix}{source_prefix}_{i + 1:02d}_{safe_title}.txt"
    if is_thin:
        logger.warning(
            f"{source_prefix} item {i + 1} is thin ({len(raw_text)} chars) "
            "— likely to be rejected by classifier"
        )
    return _make_doc(filename, raw_text)


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------

def fetch_yahoo_finance_news(ticker: str, max_articles: int = 15) -> list[dict]:
    """Fetch recent news for a ticker via yfinance (free, no key required)."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        return []

    logger.info(f"Fetching Yahoo Finance news for {ticker} (max {max_articles})")

    try:
        ticker_obj = yf.Ticker(ticker)
        news_items = ticker_obj.news or []
    except Exception as e:
        logger.error(f"yfinance fetch failed for {ticker}: {e}")
        return []

    docs = [
        _build_fetched_doc(
            _assemble_yfinance_text(item),
            (item.get("title") or f"article_{i + 1}").strip(),
            i,
            f"yf_{ticker}",
        )
        for i, item in enumerate(news_items[:max_articles])
    ]
    logger.info(f"Yahoo Finance: assembled {len(docs)} docs for {ticker}")
    return docs


# ---------------------------------------------------------------------------
# News API
# ---------------------------------------------------------------------------

def fetch_newsapi_news(query: str, api_key: str, max_articles: int = 15) -> list[dict]:
    """Fetch news via newsapi.org. Free tier truncates article body to ~200 chars."""
    try:
        from newsapi import NewsApiClient
    except ImportError:
        logger.error("newsapi-python not installed. Run: pip install newsapi-python")
        return []

    logger.info(f"Fetching News API for query='{query}' (max {max_articles})")

    try:
        client = NewsApiClient(api_key=api_key)
        response = client.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            page_size=min(max_articles, 100),
        )
    except Exception as e:
        logger.error(f"News API fetch failed for query '{query}': {e}")
        return []

    articles = (response or {}).get("articles") or []
    docs = []

    for i, article in enumerate(articles[:max_articles]):
        title = (article.get("title") or f"article_{i + 1}").strip()
        description = (article.get("description") or "").strip()
        content = (article.get("content") or "").strip()
        source_name = ((article.get("source") or {}).get("name") or "").strip()
        published_at = (article.get("publishedAt") or "")[:10]
        url = (article.get("url") or "").strip()

        parts: list[str] = []
        header_parts = [x for x in [published_at, source_name, title] if x]
        if header_parts:
            parts.append(" | ".join(header_parts))
        if description and description != title:
            parts.append(description)
        if content:
            parts.append(content)
        if url:
            parts.append(f"Source: {url}")

        docs.append(_build_fetched_doc("\n\n".join(parts), title, i, "newsapi"))

    logger.info(f"News API: assembled {len(docs)} docs for query '{query}'")
    return docs


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def fetch_live_docs(
    ticker: str,
    company_name: str,
    use_yahoo: bool = True,
    use_newsapi: bool = False,
    newsapi_key: Optional[str] = None,
    max_per_source: int = 15,
) -> list[dict]:
    """
    Fetch live documents from enabled sources in parallel (ThreadPoolExecutor).
    Returns a combined list of ingest_file-compatible dicts.
    """
    futures: dict[str, object] = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        if use_yahoo:
            futures["yahoo"] = executor.submit(
                fetch_yahoo_finance_news, ticker, max_per_source
            )
        if use_newsapi:
            key = newsapi_key or os.getenv("NEWSAPI_KEY", "")
            if not key:
                logger.error(
                    "News API enabled but no NEWSAPI_KEY found. "
                    "Provide it in the sidebar or set NEWSAPI_KEY in your .env file."
                )
            else:
                query = f'"{ticker}" OR "{company_name}"'
                futures["newsapi"] = executor.submit(
                    fetch_newsapi_news, query, key, max_per_source
                )

    all_docs: list[dict] = []
    if "yahoo" in futures:
        all_docs.extend(futures["yahoo"].result())
    if "newsapi" in futures:
        all_docs.extend(futures["newsapi"].result())

    logger.info(f"fetch_live_docs total: {len(all_docs)} docs from all sources")
    return all_docs
