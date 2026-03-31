# FinSentinel — Full Implementation Plan
## For Claude Code Sessions

---

## STEP 0 — Project Scaffolding

Run in Claude Code:

```bash
mkdir finsentinel && cd finsentinel
mkdir -p pipeline utils tests outputs

touch app.py
touch pipeline/__init__.py pipeline/ingestion.py pipeline/classification.py
touch pipeline/chunking.py pipeline/token_enforcement.py
touch pipeline/finbert.py pipeline/aggregation.py
touch utils/__init__.py utils/claude_client.py utils/logger.py utils/prompts.py
touch tests/test_ingestion.py tests/test_classification.py
touch tests/test_chunking.py tests/test_token_enforcement.py tests/test_finbert.py
touch requirements.txt .env.example .gitignore README.md CLAUDE.md
echo "outputs/" >> .gitignore
echo ".env" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "venv/" >> .gitignore
```

---

## STEP 1 — requirements.txt

```
anthropic>=0.40.0
streamlit>=1.35.0
pdfplumber>=0.11.0
python-docx>=1.1.0
transformers>=4.40.0
torch>=2.2.0
pandas>=2.2.0
openpyxl>=3.1.0
plotly>=5.20.0
python-dotenv>=1.0.0
chardet>=5.2.0
beautifulsoup4>=4.12.0
nltk>=3.8.0
tqdm>=4.66.0
```

---

## STEP 2 — .env.example

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

---

## STEP 3 — utils/logger.py

```python
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

---

## STEP 4 — utils/prompts.py

```python
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
{
  "document_type": "equity_research|10K|10Q|MDA|earnings_transcript|news|press_release|filing|other",
  "suitable_for_sentiment": true or false,
  "rejection_reason": null or "explanation string",
  "sentiment_bias_warning": null or "bullish_biased" or "bearish_biased",
  "confidence": 0.0 to 1.0
}

Document filename: {filename}
Document beginning (first 3000 characters):
---
{text_preview}
---
"""

CHUNKING_PROMPT = """You are a financial text analyst preparing passages for FinBERT sentiment analysis.

Document type: {document_type}
Filename: {filename}

Your task: Extract meaningful text passages from this document.

EXTRACT passages containing:
- Management commentary and outlook statements
- Risk factor descriptions and discussions
- Analyst opinions, ratings rationale, price target justification
- Financial performance discussion (narrative, not just numbers)
- Forward-looking statements and guidance
- News commentary and market analysis  
- Competitive positioning discussion
- Discussion of macro factors (commodity prices, regulations, etc.)

DO NOT extract:
- Document headers, page numbers, running footers
- Table of contents entries
- Legal disclaimers and standard boilerplate ("This report has been prepared by...")
- Auditor standard certification language
- Pure numerical tables without surrounding narrative
- Regulatory form field labels
- Repetitive certification or signature blocks

Each chunk must be a coherent, standalone passage of approximately 300-480 words (do not
split mid-sentence). The text should be verbatim from the document.

Respond ONLY with a valid JSON array. No text outside the JSON. No markdown. Raw JSON only.

Schema:
[
  {{
    "chunk_id": 1,
    "text": "verbatim passage from document",
    "section": "section name this came from",
    "page_hint": estimated_page_number_or_null
  }}
]

Document text:
---
{document_text}
---
"""
```

---

## STEP 5 — utils/claude_client.py

```python
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
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    return json.loads(cleaned)
```

---

## STEP 6 — pipeline/ingestion.py

```python
import io
from pathlib import Path
import chardet
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup
from utils.logger import get_logger

logger = get_logger("ingestion")

def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF using pdfplumber. Returns empty string for image-only PDFs."""
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
    return "\n\n".join(texts)

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_text_from_txt(path: str) -> str:
    raw_bytes = Path(path).read_bytes()
    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding", "utf-8") or "utf-8"
    return raw_bytes.decode(encoding, errors="replace")

def extract_text_from_html(path: str) -> str:
    raw_bytes = Path(path).read_bytes()
    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding", "utf-8") or "utf-8"
    html = raw_bytes.decode(encoding, errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def ingest_file(path: str) -> dict:
    """
    Returns:
        {filename, file_type, raw_text, char_count, is_scanned_pdf}
    """
    p = Path(path)
    suffix = p.suffix.lower()
    filename = p.name
    is_scanned_pdf = False

    if suffix == ".pdf":
        raw_text = extract_text_from_pdf(path)
        if len(raw_text.strip()) < 200:
            is_scanned_pdf = True
            logger.warning(f"{filename}: Very little text extracted — may be scanned PDF")
        file_type = "pdf"
    elif suffix in (".docx", ".doc"):
        raw_text = extract_text_from_docx(path)
        file_type = "docx"
    elif suffix in (".txt", ".md"):
        raw_text = extract_text_from_txt(path)
        file_type = "txt"
    elif suffix in (".html", ".htm"):
        raw_text = extract_text_from_html(path)
        file_type = "html"
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return {
        "filename": filename,
        "file_type": file_type,
        "raw_text": raw_text,
        "char_count": len(raw_text),
        "is_scanned_pdf": is_scanned_pdf,
    }
```

---

## STEP 7 — pipeline/classification.py

```python
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
```

---

## STEP 8 — pipeline/chunking.py

```python
from dataclasses import dataclass
from typing import Optional
from utils.prompts import CHUNKING_PROMPT
from utils.claude_client import call_claude_json
from utils.logger import get_logger

logger = get_logger("chunking")

MAX_CHARS_PER_CALL = 80_000  # ~60k tokens, safe for Claude

@dataclass
class ChunkResult:
    chunk_id: int
    text: str
    section: str
    page_hint: Optional[int]

def extract_chunks(filename: str, raw_text: str, document_type: str) -> list[ChunkResult]:
    """Extract meaningful chunks from a document. Handles large docs via windowing."""
    all_chunks = []
    
    if len(raw_text) <= MAX_CHARS_PER_CALL:
        windows = [raw_text]
    else:
        # Split into overlapping windows to avoid cutting mid-thought
        stride = MAX_CHARS_PER_CALL - 2000  # 2000 char overlap
        windows = [raw_text[i:i+MAX_CHARS_PER_CALL] for i in range(0, len(raw_text), stride)]
        logger.info(f"{filename}: Large doc, processing in {len(windows)} windows")

    global_chunk_id = 1
    seen_texts = set()

    for window_idx, window_text in enumerate(windows):
        prompt = CHUNKING_PROMPT.format(
            document_type=document_type,
            filename=filename,
            document_text=window_text
        )
        logger.info(f"{filename}: Extracting chunks (window {window_idx+1}/{len(windows)})")
        
        raw_chunks = call_claude_json(prompt, max_tokens=8000)
        
        for c in raw_chunks:
            text = c.get("text", "").strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            all_chunks.append(ChunkResult(
                chunk_id=global_chunk_id,
                text=text,
                section=c.get("section", "Unknown"),
                page_hint=c.get("page_hint"),
            ))
            global_chunk_id += 1

    logger.info(f"{filename}: Extracted {len(all_chunks)} unique chunks")
    return all_chunks
```

---

## STEP 9 — pipeline/token_enforcement.py

```python
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
    current_sentences = []
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
```

---

## STEP 10 — pipeline/finbert.py

```python
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
        batch = texts[i:i+batch_size]
        batch_ids = chunk_ids[i:i+batch_size]

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
```

---

## STEP 11 — pipeline/aggregation.py

```python
import pandas as pd
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("aggregation")

def build_results_dataframe(
    doc_records: list[dict],  # from ingestion + classification
    chunk_records: list[dict],  # enriched chunks
    sentiment_results: list,   # SentimentResult list
) -> pd.DataFrame:
    """Build the master results DataFrame with all chunk-level data."""
    
    sentiment_map = {s.chunk_id: s for s in sentiment_results}
    rows = []

    for doc in doc_records:
        for chunk, token_count in doc["chunks"]:
            sentiment = sentiment_map.get(chunk.chunk_id)
            if not sentiment:
                continue
            rows.append({
                "doc_id": doc["doc_id"],
                "filename": doc["filename"],
                "file_type": doc["file_type"],
                "document_type": doc["document_type"],
                "bias_warning": doc.get("bias_warning"),
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "page_hint": chunk.page_hint,
                "token_count": token_count,
                "text": chunk.text,
                "positive": sentiment.positive,
                "negative": sentiment.negative,
                "neutral": sentiment.neutral,
                "sentiment_label": sentiment.label,
                "sentiment_confidence": sentiment.confidence,
                "net_sentiment": sentiment.positive - sentiment.negative,
            })

    return pd.DataFrame(rows)

def compute_document_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment metrics per document."""
    summary = df.groupby(["doc_id", "filename", "document_type"]).agg(
        chunk_count=("chunk_id", "count"),
        mean_positive=("positive", "mean"),
        mean_negative=("negative", "mean"),
        mean_neutral=("neutral", "mean"),
        mean_net_sentiment=("net_sentiment", "mean"),
        std_net_sentiment=("net_sentiment", "std"),
        pct_positive=("sentiment_label", lambda x: (x == "positive").mean() * 100),
        pct_negative=("sentiment_label", lambda x: (x == "negative").mean() * 100),
        pct_neutral=("sentiment_label", lambda x: (x == "neutral").mean() * 100),
    ).reset_index()
    
    summary = summary.sort_values("mean_net_sentiment", ascending=False)
    return summary

def compute_portfolio_metrics(summary_df: pd.DataFrame) -> dict:
    """Compute aggregate metrics across all documents."""
    if summary_df.empty:
        return {}
    
    return {
        "overall_net_sentiment": summary_df["mean_net_sentiment"].mean(),
        "total_chunks_analyzed": summary_df["chunk_count"].sum(),
        "total_docs_analyzed": len(summary_df),
        "pct_positive_docs": (summary_df["mean_net_sentiment"] > 0.05).mean() * 100,
        "pct_negative_docs": (summary_df["mean_net_sentiment"] < -0.05).mean() * 100,
        "most_bullish_doc": summary_df.iloc[0]["filename"] if len(summary_df) > 0 else None,
        "most_bearish_doc": summary_df.iloc[-1]["filename"] if len(summary_df) > 0 else None,
    }

def export_to_excel(
    full_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    rejected_docs: list[dict],
    output_path: str,
):
    rejected_df = pd.DataFrame(rejected_docs)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Document Summary", index=False)
        full_df.drop(columns=["text"], errors="ignore").to_excel(
            writer, sheet_name="All Chunks (no text)", index=False
        )
        full_df.to_excel(writer, sheet_name="All Chunks (with text)", index=False)
        if not rejected_df.empty:
            rejected_df.to_excel(writer, sheet_name="Rejected Documents", index=False)
    logger.info(f"Exported results to {output_path}")
```

---

## STEP 12 — app.py (Streamlit UI)

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
from pathlib import Path

from pipeline.ingestion import ingest_file
from pipeline.classification import classify_document
from pipeline.chunking import extract_chunks
from pipeline.token_enforcement import enforce_token_limit
from pipeline.finbert import get_finbert, run_finbert
from pipeline.aggregation import (
    build_results_dataframe,
    compute_document_summary,
    compute_portfolio_metrics,
    export_to_excel,
)

st.set_page_config(
    page_title="FinSentinel — Financial Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
)

# Pre-load FinBERT (cached)
@st.cache_resource
def load_finbert():
    return get_finbert()

st.title("📊 FinSentinel")
st.caption("Financial Document Sentiment Analyzer — powered by Claude + FinBERT")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    finbert_batch_size = st.slider("FinBERT batch size", 4, 32, 16)
    show_rejected = st.checkbox("Show rejected documents", value=True)
    st.info("Upload financial documents (PDF, DOCX, TXT, HTML) and click **Run Analysis**.")

uploaded_files = st.file_uploader(
    "Upload documents",
    accept_multiple_files=True,
    type=["pdf", "docx", "doc", "txt", "html", "htm"],
)

if not uploaded_files:
    st.info("Upload one or more financial documents to begin.")
    st.stop()

st.write(f"**{len(uploaded_files)} file(s) ready.** Click Run Analysis to start the pipeline.")

if st.button("🚀 Run Analysis", type="primary"):
    load_finbert()  # ensure model is warm

    doc_records = []
    rejected_docs = []
    all_sentiment_results = []
    doc_id_counter = 0

    progress = st.progress(0)
    status = st.status("Starting pipeline...", expanded=True)

    with tempfile.TemporaryDirectory() as tmpdir:

        # Phase 1+2+3: Ingest, Classify, Chunk
        for i, uploaded_file in enumerate(uploaded_files):
            status.write(f"📄 Processing: {uploaded_file.name}")
            progress.progress((i) / (len(uploaded_files) * 2))

            # Save to disk
            tmp_path = os.path.join(tmpdir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Ingest
            try:
                ingested = ingest_file(tmp_path)
            except Exception as e:
                rejected_docs.append({
                    "filename": uploaded_file.name,
                    "rejection_reason": f"Ingestion error: {e}",
                    "document_type": "unknown",
                })
                continue

            if ingested["is_scanned_pdf"]:
                rejected_docs.append({
                    "filename": uploaded_file.name,
                    "rejection_reason": "Scanned/image PDF — no extractable text",
                    "document_type": "pdf",
                })
                continue

            # Classify
            classification = classify_document(ingested["filename"], ingested["raw_text"])
            if not classification.suitable_for_sentiment:
                rejected_docs.append({
                    "filename": uploaded_file.name,
                    "rejection_reason": classification.rejection_reason,
                    "document_type": classification.document_type,
                })
                continue

            # Chunk
            raw_chunks = extract_chunks(
                ingested["filename"],
                ingested["raw_text"],
                classification.document_type,
            )

            # Token enforce
            enforced_chunks = enforce_token_limit(raw_chunks)

            doc_id_counter += 1
            doc_records.append({
                "doc_id": doc_id_counter,
                "filename": ingested["filename"],
                "file_type": ingested["file_type"],
                "document_type": classification.document_type,
                "bias_warning": classification.sentiment_bias_warning,
                "chunks": enforced_chunks,
            })

        # Phase 4: FinBERT
        status.write("🤖 Running FinBERT sentiment analysis...")
        all_chunks_flat = []
        for doc in doc_records:
            all_chunks_flat.extend(doc["chunks"])

        if not all_chunks_flat:
            st.error("No usable chunks found. All documents were rejected or empty.")
            st.stop()

        finbert_progress = st.progress(0)

        def update_finbert_progress(done, total):
            finbert_progress.progress(done / total)

        all_sentiment_results = run_finbert(
            all_chunks_flat,
            batch_size=finbert_batch_size,
            progress_callback=update_finbert_progress,
        )

        # Phase 5: Aggregate
        status.write("📊 Computing metrics...")
        full_df = build_results_dataframe(doc_records, all_chunks_flat, all_sentiment_results)
        summary_df = compute_document_summary(full_df)
        portfolio = compute_portfolio_metrics(summary_df)

        # Export
        output_path = os.path.join(tmpdir, "finsentinel_results.xlsx")
        export_to_excel(full_df, summary_df, rejected_docs, output_path)
        with open(output_path, "rb") as f:
            excel_bytes = f.read()

        status.update(label="✅ Analysis complete!", state="complete")

    # --- RESULTS UI ---
    st.divider()
    st.header("Results")

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    net = portfolio.get("overall_net_sentiment", 0)
    col1.metric("Overall Net Sentiment", f"{net:.3f}", delta=f"{'Bullish' if net > 0.05 else 'Bearish' if net < -0.05 else 'Neutral'}")
    col2.metric("Docs Analyzed", portfolio.get("total_docs_analyzed", 0))
    col3.metric("Chunks Analyzed", portfolio.get("total_chunks_analyzed", 0))
    col4.metric("Docs Rejected", len(rejected_docs))

    st.subheader("Net Sentiment by Document")
    fig_bar = px.bar(
        summary_df,
        x="filename",
        y="mean_net_sentiment",
        color="mean_net_sentiment",
        color_continuous_scale=["#d62728", "#aaaaaa", "#2ca02c"],
        color_continuous_midpoint=0,
        labels={"mean_net_sentiment": "Net Sentiment (Pos − Neg)", "filename": "Document"},
        title="Net Sentiment Score per Document",
    )
    fig_bar.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_bar, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Sentiment Distribution (All Chunks)")
        label_counts = full_df["sentiment_label"].value_counts()
        fig_pie = px.pie(
            values=label_counts.values,
            names=label_counts.index,
            color=label_counts.index,
            color_discrete_map={"positive": "#2ca02c", "negative": "#d62728", "neutral": "#aaaaaa"},
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Document Summary Table")
        st.dataframe(
            summary_df[["filename", "document_type", "chunk_count", "mean_net_sentiment",
                         "std_net_sentiment", "pct_positive", "pct_negative", "pct_neutral"]].round(3),
            use_container_width=True,
        )

    st.subheader("Full Results (All Chunks)")
    st.dataframe(
        full_df[["filename", "document_type", "section", "sentiment_label",
                 "positive", "negative", "neutral", "net_sentiment", "token_count"]].round(3),
        use_container_width=True,
    )

    if show_rejected and rejected_docs:
        with st.expander(f"⚠️ Rejected Documents ({len(rejected_docs)})"):
            st.dataframe(pd.DataFrame(rejected_docs), use_container_width=True)

    # Downloads
    st.divider()
    st.subheader("Downloads")
    col_d1, col_d2 = st.columns(2)
    col_d1.download_button(
        "📥 Download Full Results (Excel)",
        data=excel_bytes,
        file_name="finsentinel_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    col_d2.download_button(
        "📥 Download Summary CSV",
        data=summary_df.to_csv(index=False),
        file_name="finsentinel_summary.csv",
        mime="text/csv",
    )
```

---

## STEP 13 — README.md

```markdown
# FinSentinel — Financial Document Sentiment Analyzer

Upload equity research reports, 10-K/Q filings, MD&A sections, and news articles.
FinSentinel classifies, filters, and chunks each document using Claude, then runs 
FinBERT to produce chunk-level and document-level sentiment scores.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
streamlit run app.py
```

## Supported File Types
- PDF (.pdf)
- Word (.docx)
- Plain text (.txt)
- HTML (.html)

## Output
- Net sentiment score per document (positive − negative)
- Chunk-level sentiment breakdown
- Downloadable Excel with all results
- Rejected documents log with reason

## Notes
- Scanned PDFs (image-only) are not supported — OCR is not included
- Large documents (>80k characters) are processed in overlapping windows
- FinBERT max input is 512 BERT tokens; chunks are enforced at this limit
```

---

## TESTING CHECKLIST

Run before any demo:
```bash
# Unit tests
python -m pytest tests/ -v

# Smoke test with one real PDF
python -c "
from pipeline.ingestion import ingest_file
from pipeline.classification import classify_document
r = ingest_file('sample.pdf')
c = classify_document(r['filename'], r['raw_text'])
print(c)
"

# FinBERT sanity check
python -c "
from pipeline.finbert import get_finbert
model = get_finbert()
result = model(['Revenue increased significantly year over year.', 'The company faces severe bankruptcy risk.'])
print(result)
"
```

---

## CLAUDE CODE WORKFLOW

When Claude Code starts a new session on this project:
1. Read `CLAUDE.md` first
2. Check the Memory Log at the bottom — what phases are done?
3. Pick up from the first unchecked phase
4. After completing a phase, append to the Memory Log in CLAUDE.md
5. Run the relevant unit tests before marking complete
6. Never skip the token enforcement step — FinBERT will error on >512 tokens
