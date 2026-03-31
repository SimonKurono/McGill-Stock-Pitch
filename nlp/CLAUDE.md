# CLAUDE.md — FinSentinel: Financial Document Sentiment Analyzer
# Living memory file. Claude must read this at session start and update it after every meaningful step.

---

## PROJECT OVERVIEW

**Name:** FinSentinel  
**Purpose:** Upload financial documents (equity research, 10-K/Q, MD&A, news, filings) and run an agentic pipeline that classifies, filters, chunks, and scores each document using FinBERT. Outputs an interactive dashboard and downloadable results dataframe.

**Primary Use Case:** Stock pitch preparation (initial target: Skeena Resources, TSX: SKE)

---

## TECHNOLOGY STACK

| Layer | Technology | Reason |
|---|---|---|
| UI | Streamlit | Fast iteration, native file upload, dataframe rendering |
| Document ingestion | pdfplumber (PDF), python-docx (DOCX), chardet (TXT/HTML encoding) | Best text fidelity for financial PDFs |
| Agent — classify & chunk | Anthropic Claude API (`claude-sonnet-4-20250514`) | Structured JSON output, document-aware reasoning |
| Sentiment model | `ProsusAI/finbert` via HuggingFace Transformers | Purpose-built for financial text |
| Data layer | pandas + openpyxl | Tabular aggregation and Excel export |
| Token counting | `transformers.AutoTokenizer` (bert-base-uncased) | Match FinBERT's tokenizer exactly |
| Config | python-dotenv | API key management |

---

## ARCHITECTURE

```
[User uploads files via Streamlit]
         │
         ▼
[1. INGESTION LAYER]
   - Extract raw text per document
   - Detect file type (PDF / DOCX / TXT / HTML)
   - Store: {filename, file_type, raw_text, page_count}
         │
         ▼
[2. CLASSIFICATION AGENT — Claude API]
   - Input: first ~3000 chars of raw_text + filename
   - Output (JSON):
       {
         "document_type": "<equity_research|10K|10Q|MDA|news|press_release|filing|other>",
         "suitable_for_sentiment": <true|false>,
         "rejection_reason": "<null or string>",
         "sentiment_bias_warning": "<null|bullish_biased|bearish_biased>",
         "confidence": <0.0–1.0>
       }
   - Rejection criteria:
       • Boilerplate-only docs (pure legal disclaimers, index pages)
       • Documents with >70% neutral/administrative text
       • Extremely one-sided promotional material (e.g. company press kit cover pages)
         │
         ▼  (unsuitable docs are logged and skipped)
         ▼
[3. CHUNK EXTRACTION AGENT — Claude API]
   - Input: full raw_text, document_type
   - Task: Extract semantically important passages only
       • Include: outlook statements, risk factors, financial commentary,
                  analyst opinions, guidance, mgmt discussion, news commentary
       • Exclude: table of contents, headers, footnotes, legal disclaimers,
                  auditor boilerplate, repetitive regulatory language
   - Output (JSON array):
       [
         {"chunk_id": 1, "text": "...", "section": "Risk Factors", "page_hint": 12},
         ...
       ]
   - Each chunk: max 512 BERT tokens (enforced post-hoc by tokenizer trim)
   - Claude should aim for ~400–480 tokens per chunk to leave headroom
         │
         ▼
[4. TOKEN ENFORCEMENT LAYER — Python]
   - Load bert-base-uncased tokenizer
   - For each chunk: tokenize → if > 512 tokens, split at sentence boundary
   - Final chunks stored in master DataFrame:
       columns: [doc_id, filename, document_type, chunk_id, section, text, token_count]
         │
         ▼
[5. FINBERT INFERENCE]
   - Model: ProsusAI/finbert
   - Run in batches of 16 (GPU if available, else CPU)
   - Per chunk output:
       {positive_score, negative_score, neutral_score, predicted_label, confidence}
   - Progress bar shown in Streamlit
         │
         ▼
[6. AGGREGATION & METRICS]
   Per document:
     - mean_positive, mean_negative, mean_neutral
     - net_sentiment_score = mean_positive - mean_negative
     - sentiment_label (majority vote)
     - std_sentiment, chunk_count
   
   Portfolio-level (all docs):
     - overall_net_sentiment
     - sentiment distribution (positive/neutral/negative %)
     - most bullish document, most bearish document
     - sentiment by document_type breakdown
         │
         ▼
[7. OUTPUT]
   Streamlit displays:
     - Summary metrics panel
     - Bar chart: net sentiment per document
     - Distribution chart: positive/neutral/negative across all chunks
     - Full results dataframe (sortable/filterable)
     - Rejected documents table with reasons
   
   Downloads:
     - results_full.xlsx  (all chunks with scores)
     - results_summary.xlsx  (per-document aggregation)
     - rejected_docs.csv
```

---

## FILE STRUCTURE

```
finsentinel/
├── CLAUDE.md                    ← THIS FILE (update constantly)
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── app.py                       ← Streamlit entrypoint
│
├── pipeline/
│   ├── __init__.py
│   ├── ingestion.py             ← Raw text extraction per file type
│   ├── classification.py        ← Claude agent: classify + filter docs
│   ├── chunking.py              ← Claude agent: extract meaningful chunks
│   ├── token_enforcement.py     ← Tokenizer-based chunk size validation/splitting
│   ├── finbert.py               ← FinBERT inference wrapper
│   └── aggregation.py           ← Metrics computation and DataFrame assembly
│
├── utils/
│   ├── __init__.py
│   ├── claude_client.py         ← Anthropic API client wrapper with retry logic
│   ├── logger.py                ← Structured logging
│   └── prompts.py               ← All Claude prompt templates (single source of truth)
│
├── outputs/                     ← Generated output files (gitignored)
│   └── .gitkeep
│
└── tests/
    ├── test_ingestion.py
    ├── test_classification.py
    ├── test_chunking.py
    ├── test_token_enforcement.py
    └── test_finbert.py
```

---

## IMPLEMENTATION PHASES

### Phase 0 — Environment Setup
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Install dependencies from requirements.txt
- [ ] Create `.env` with `ANTHROPIC_API_KEY`
- [ ] Verify FinBERT loads: `python -c "from transformers import pipeline; pipeline('text-classification', model='ProsusAI/finbert')"`

### Phase 1 — Ingestion Layer (`pipeline/ingestion.py`)
- [ ] `extract_text_from_pdf(path) -> str` using pdfplumber
- [ ] `extract_text_from_docx(path) -> str` using python-docx
- [ ] `extract_text_from_txt(path) -> str` with chardet encoding detection
- [ ] `extract_text_from_html(path) -> str` using BeautifulSoup (strip tags)
- [ ] `ingest_file(path) -> dict` dispatcher returning {filename, file_type, raw_text, char_count}
- [ ] Unit test: test each file type with a sample doc

### Phase 2 — Classification Agent (`pipeline/classification.py`)
- [ ] Write `CLASSIFICATION_PROMPT` in `utils/prompts.py`
- [ ] `classify_document(filename, raw_text_preview) -> ClassificationResult`
- [ ] ClassificationResult dataclass: document_type, suitable, rejection_reason, bias_warning, confidence
- [ ] Implement retry logic (max 3 attempts, exponential backoff) in claude_client.py
- [ ] Unit test: known-suitable doc, known-reject doc (e.g. a pure disclaimer page)

### Phase 3 — Chunk Extraction Agent (`pipeline/chunking.py`)
- [ ] Write `CHUNKING_PROMPT` in `utils/prompts.py`
- [ ] `extract_chunks(filename, raw_text, document_type) -> list[ChunkResult]`
- [ ] ChunkResult dataclass: chunk_id, text, section, page_hint
- [ ] Handle large docs: if raw_text > 80k chars, split into windows and call Claude per window, dedup
- [ ] Unit test: verify no headers/disclaimers present in output chunks

### Phase 4 — Token Enforcement (`pipeline/token_enforcement.py`)
- [ ] Load `bert-base-uncased` tokenizer once at module level
- [ ] `enforce_token_limit(chunks: list[ChunkResult], max_tokens=512) -> list[ChunkResult]`
- [ ] If chunk > 512 tokens: split at last sentence boundary before limit
- [ ] Add `token_count` field to each chunk
- [ ] Unit test: feed a 700-token chunk, verify output chunks all ≤ 512

### Phase 5 — FinBERT Inference (`pipeline/finbert.py`)
- [ ] Load `ProsusAI/finbert` pipeline at startup (cache with `@st.cache_resource` in app.py)
- [ ] `run_finbert(chunks: list[ChunkResult], batch_size=16) -> list[SentimentResult]`
- [ ] SentimentResult: chunk_id, positive, negative, neutral, label, confidence
- [ ] Progress callback for Streamlit progress bar
- [ ] Unit test: "Revenue increased significantly" → positive; "Bankruptcy risk" → negative

### Phase 6 — Aggregation (`pipeline/aggregation.py`)
- [ ] `build_results_dataframe(doc_metadata, chunks, sentiments) -> pd.DataFrame`
- [ ] `compute_document_summary(df) -> pd.DataFrame`
- [ ] `compute_portfolio_metrics(summary_df) -> dict`
- [ ] `export_to_excel(full_df, summary_df, rejected_docs, output_path)`

### Phase 7 — Streamlit UI (`app.py`)
- [ ] Page config, title, sidebar
- [ ] File uploader (accept: pdf, docx, txt, html; multiple files)
- [ ] "Run Analysis" button with spinner
- [ ] Metrics panel: net sentiment score, % positive, % negative, % neutral
- [ ] Bar chart: net sentiment per document (plotly or altair)
- [ ] Pie/donut: overall distribution
- [ ] Full results dataframe with st.dataframe (searchable)
- [ ] Rejected documents expander
- [ ] Download buttons: full Excel, summary Excel, rejected CSV

### Phase 8 — Polish & Testing
- [ ] End-to-end test with real SKE documents
- [ ] Handle edge cases: empty PDFs, scanned PDFs (warn user, no OCR)
- [ ] Add processing time estimates to UI
- [ ] README with setup instructions

---

## PROMPT ENGINEERING SPECS

### Classification Prompt
```
You are a financial document analyst. Given the beginning of a document, classify it and determine 
if it is suitable for financial sentiment analysis.

Suitable documents: equity research reports, 10-K, 10-Q, MD&A sections, earnings call transcripts,
financial news articles, analyst commentary, company press releases with substance.

NOT suitable: pure legal disclaimers (>80% legal boilerplate), index/table of contents only pages,
regulatory forms with no narrative text, cover pages, glossaries.

Also flag if a document is extremely biased (e.g. pure promotional material with no critical analysis).

Respond ONLY with valid JSON. No explanation outside the JSON.
Schema:
{
  "document_type": "equity_research|10K|10Q|MDA|earnings_transcript|news|press_release|filing|other",
  "suitable_for_sentiment": true|false,
  "rejection_reason": null or "string explaining why rejected",
  "sentiment_bias_warning": null|"bullish_biased"|"bearish_biased",
  "confidence": 0.0-1.0
}
```

### Chunking Prompt
```
You are a financial text analyst. Your job is to extract meaningful text passages from a financial 
document for sentiment analysis using FinBERT.

Document type: {document_type}

EXTRACT chunks that contain:
- Management commentary and outlook statements
- Risk factor descriptions
- Analyst opinions and ratings rationale  
- Financial performance discussion (not just raw numbers)
- Forward-looking statements
- News commentary and market analysis
- Competitive positioning discussion

DO NOT extract:
- Document headers, page numbers, footers
- Table of contents entries
- Legal disclaimers and boilerplate
- Auditor standard language
- Pure numerical tables (without narrative context)
- Regulatory filing form fields
- Repetitive certification language

Each chunk should be a coherent passage of 300-480 words. Return ONLY valid JSON array.
No text outside the JSON.

Schema:
[
  {
    "chunk_id": 1,
    "text": "the extracted passage...",
    "section": "name of section this came from",
    "page_hint": estimated page number or null
  }
]
```

---

## KEY DESIGN DECISIONS & RATIONALE

| Decision | Choice | Why |
|---|---|---|
| Chunk size | Max 512 BERT tokens | FinBERT hard limit; we target 400-480 to be safe |
| Classification preview length | First 3000 chars | Enough to identify doc type without burning tokens |
| Chunking model | Claude (not rule-based) | Financial docs have irregular structure; LLM understands context |
| FinBERT vs VADER/TextBlob | FinBERT | Purpose-trained on financial phraseology; "strong headwinds" = negative (VADER would miss this) |
| Batch size for FinBERT | 16 | Balance between memory and throughput on CPU |
| UI framework | Streamlit | Zero frontend boilerplate; native file upload; ideal for data tools |
| Export format | Excel (.xlsx) | Most accessible for analysts; supports multiple sheets |

---

## KNOWN LIMITATIONS & MITIGATIONS

| Limitation | Mitigation |
|---|---|
| Scanned PDFs (image-only) | Detect with pdfplumber text length = 0, warn user, skip gracefully |
| Very large docs (>200 pages) | Window the raw text into 80k char chunks for Claude API; reassemble |
| FinBERT CPU speed | ~2-5 sec per chunk on CPU; show progress bar; GPU auto-detected |
| Claude API rate limits | Exponential backoff retry in claude_client.py (max 3 retries, 2/4/8s) |
| Token counting mismatch | Use BERT tokenizer (not tiktoken) to count — must match FinBERT exactly |
| Biased source docs | Classification agent flags bias_warning; user sees this in UI |

---

## DEPENDENCIES (requirements.txt)

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

## MEMORY LOG
> Claude must append an entry here after completing each phase.
> Format: `[PHASE X COMPLETE — date] Notes`

- [PROJECT INITIALIZED] CLAUDE.md created. Awaiting Phase 0 kickoff.
- [PHASES 0–8 COMPLETE — 2026-03-30] Full implementation delivered in finsentinel/. All pipeline modules created: ingestion, classification, chunking, token_enforcement, finbert, aggregation. Streamlit UI (app.py) wired end-to-end. requirements.txt, .env.example, .gitignore, README.md in place. Next: install deps, set ANTHROPIC_API_KEY in .env, run `streamlit run app.py`.

---

## ENVIRONMENT

- Python: 3.11+
- ANTHROPIC_API_KEY: set in .env (never commit)
- Optional: CUDA-compatible GPU for faster FinBERT inference
- Tested on: macOS / Linux (Windows compatible with path adjustments)

---

## QUICK START (for Claude to follow)

```bash
# 1. Clone / navigate to project directory
cd finsentinel

# 2. Create virtual env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...

# 5. Run app
streamlit run app.py
```

---

## NOTES FOR STOCK PITCH USE (SKE / Skeena Resources)

Recommended document types to upload:
1. Skeena 2023 Annual Report / AIF
2. Latest MD&A (most recent quarter)
3. Equity research reports from BMO, Canaccord, Scotiabank etc.
4. Recent news articles about SKE, gold/silver prices, BC mining
5. Comparable company reports (Seabridge Gold, Brixton Metals)
6. Technical report (NI 43-101) — **flag**: likely high neutral score due to technical language

Expected output: net_sentiment_score per document, overall portfolio sentiment, 
which analyst reports are most bullish/bearish, which risk sections score most negative.
