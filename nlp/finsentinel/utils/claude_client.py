import os
import time
import json
import anthropic
from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()
logger = get_logger("claude_client")

_client = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def call_claude(prompt: str, max_tokens: int = 4096, retries: int = 3) -> str:
    """Call Claude API with exponential backoff retry."""
    client = get_client()
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            wait = 2 ** attempt
            logger.warning(f"Rate limit hit. Retrying in {wait}s (attempt {attempt+1}/{retries})")
            time.sleep(wait)
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Claude API call failed after all retries")


def call_claude_json(prompt: str, max_tokens: int = 4096) -> dict | list:
    """Call Claude and parse response as JSON. Strips markdown fences if present."""
    raw = call_claude(prompt, max_tokens=max_tokens)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Strip opening fence
        cleaned = cleaned[3:]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        # Strip closing fence
        if "```" in cleaned:
            cleaned = cleaned[:cleaned.rfind("```")]
    cleaned = cleaned.strip()
    return json.loads(cleaned)
