import pandas as pd
from utils.logger import get_logger

logger = get_logger("aggregation")


def build_results_dataframe(
    doc_records: list[dict],      # from ingestion + classification
    chunk_records: list[tuple],   # flat list of (ChunkResult, token_count)
    sentiment_results: list,      # SentimentResult list
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
    summary = df.groupby(["doc_id", "filename", "document_type", "file_type"]).agg(
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
        "total_chunks_analyzed": int(summary_df["chunk_count"].sum()),
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
