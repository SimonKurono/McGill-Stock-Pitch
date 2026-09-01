# FinSentinel: Financial Document Sentiment Analyzer

Upload equity research reports, 10-K/Q filings, MD&A sections, and news articles.
FinSentinel classifies, filters, and chunks each document using Claude, then runs
FinBERT to produce chunk-level and document-level sentiment scores.

## Setup

```bash
cd finsentinel
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
streamlit run app.py
```

## Supported File Types

| Format | Extension |
|--------|-----------|
| PDF    | `.pdf`    |
| Word   | `.docx`   |
| Text   | `.txt`    |
| HTML   | `.html`   |

## Pipeline

1. **Ingestion**: extracts raw text per file type
2. **Classification** (Claude): identifies document type, filters unsuitable docs
3. **Chunking** (Claude): extracts semantically meaningful passages
4. **Token enforcement**: splits any chunk exceeding 512 BERT tokens
5. **FinBERT inference**: scores each chunk: positive / negative / neutral
6. **Aggregation**: computes per-document and portfolio-level metrics

## Output

- Net sentiment score per document (positive − negative)
- Chunk-level sentiment breakdown with section labels
- Interactive bar chart and pie chart in browser
- Downloadable Excel workbook (3 sheets: summary, chunks, rejected)
- Rejected documents log with reason

## Notes

- **Scanned PDFs** (image-only) are not supported — OCR is not included
- **Large documents** (>80k characters) are processed in overlapping windows
- FinBERT max input is 512 BERT tokens; chunks are enforced at this limit
- The Claude classification agent flags bias warnings for purely promotional material
- GPU is auto-detected for faster FinBERT inference; CPU fallback is supported
