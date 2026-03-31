import streamlit as st
import pandas as pd
import plotly.express as px
import tempfile
import os

from pipeline.ingestion import ingest_file
from pipeline.classification import classify_document
from pipeline.chunking import extract_chunks
from pipeline.token_enforcement import enforce_token_limit
from pipeline.finbert import get_finbert, run_finbert
from sentence_transformers import SentenceTransformer
from pipeline.data_sources import fetch_live_docs
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


@st.cache_resource
def load_finbert():
    return get_finbert()


@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Preload both models on page open so they're warm before the user clicks Run Analysis
load_finbert()
load_sentence_model()


def _process_doc(ingested: dict, status, rejected_docs: list, sentence_model=None) -> dict | None:
    """Classify, chunk, and enforce tokens on a pre-ingested document.
    Returns a doc_record dict (without doc_id) or None if rejected/failed.
    Appends rejection details to rejected_docs in-place.
    """
    filename = ingested["filename"]

    try:
        classification = classify_document(filename, ingested["raw_text"])
    except Exception as e:
        rejected_docs.append({
            "filename": filename,
            "rejection_reason": f"Classification error: {e}",
            "document_type": "unknown",
        })
        return None

    if not classification.suitable_for_sentiment:
        rejected_docs.append({
            "filename": filename,
            "rejection_reason": classification.rejection_reason or "Unsuitable document type",
            "document_type": classification.document_type,
        })
        status.write(f"  ⚠️ Rejected: {filename} — {classification.rejection_reason}")
        return None

    status.write(
        f"  ✅ Classified as **{classification.document_type}** "
        f"(confidence: {classification.confidence:.0%})"
        + (f" — ⚠️ {classification.sentiment_bias_warning}" if classification.sentiment_bias_warning else "")
    )

    try:
        raw_chunks = extract_chunks(filename, ingested["raw_text"], classification.document_type, model=sentence_model)
    except Exception as e:
        rejected_docs.append({
            "filename": filename,
            "rejection_reason": f"Chunking error: {e}",
            "document_type": classification.document_type,
        })
        return None

    enforced_chunks = enforce_token_limit(raw_chunks)
    status.write(f"  📝 Extracted **{len(enforced_chunks)} chunks**")

    return {
        "filename": filename,
        "file_type": ingested["file_type"],
        "document_type": classification.document_type,
        "bias_warning": classification.sentiment_bias_warning,
        "chunks": enforced_chunks,
    }


st.title("📊 FinSentinel")
st.caption("Financial Document Sentiment Analyzer — powered by Claude + FinBERT")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    finbert_batch_size = st.slider("FinBERT batch size", 4, 32, 16)
    confidence_threshold = st.slider(
        "Min FinBERT confidence",
        min_value=0.0, max_value=0.95, value=0.5, step=0.05,
        help="Filter out low-confidence predictions. Higher = stricter, fewer but clearer signals.",
    )
    show_rejected = st.checkbox("Show rejected documents", value=True)
    st.divider()
    st.info(
        "Upload financial documents (PDF, DOCX, TXT, HTML) and click **Run Analysis**.\n\n"
        "**Supported:** Equity research, 10-K/Q, MD&A, press releases, earnings transcripts, "
        "news articles.\n\nUp to 50 uploaded files recommended for best performance."
    )
    st.divider()

    with st.expander("🌐 Live Data Sources", expanded=False):
        live_enabled = st.toggle("Enable live data fetching", value=False)

        live_ticker = st.text_input(
            "Ticker symbol",
            value="SKE",
            disabled=not live_enabled,
            help="e.g. SKE, AAPL, TSLA",
        )
        live_company = st.text_input(
            "Company name",
            value="Skeena Resources",
            disabled=not live_enabled,
            help="Used as the News API search query alongside the ticker",
        )

        st.caption("Sources")
        use_yahoo = st.checkbox(
            "Yahoo Finance (free, no key required)",
            value=True,
            disabled=not live_enabled,
        )
        use_newsapi = st.checkbox(
            "News API (requires NEWSAPI_KEY)",
            value=False,
            disabled=not live_enabled,
        )
        newsapi_key_input = st.text_input(
            "NEWSAPI_KEY",
            type="password",
            placeholder="Leave blank to use NEWSAPI_KEY env var",
            disabled=not (live_enabled and use_newsapi),
        )
        max_live_articles = st.slider(
            "Max articles per source",
            min_value=5,
            max_value=50,
            value=15,
            disabled=not live_enabled,
        )

# ---------------------------------------------------------------------------
# File uploader
# ---------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload documents (up to 50 files recommended)",
    accept_multiple_files=True,
    type=["pdf", "docx", "doc", "txt", "html", "htm"],
    help="Supports PDF, DOCX, TXT, and HTML. Keep to ≤50 files for best performance.",
)

has_uploads = bool(uploaded_files)
has_live = live_enabled and (use_yahoo or use_newsapi)

if not has_uploads and not has_live:
    st.info(
        "Upload one or more financial documents, "
        "or enable **Live Data Sources** in the sidebar to fetch news automatically."
    )
    st.stop()

status_parts = []
if uploaded_files:
    status_parts.append(f"**{len(uploaded_files)} uploaded file(s)**")
if live_enabled:
    live_labels = [s for s, on in [("Yahoo Finance", use_yahoo), ("News API", use_newsapi)] if on]
    if live_labels:
        status_parts.append(f"live data from {', '.join(live_labels)}")

st.write(" + ".join(status_parts) + " ready. Click **Run Analysis** to start the pipeline.")

# ---------------------------------------------------------------------------
# Run Analysis
# ---------------------------------------------------------------------------
if st.button("🚀 Run Analysis", type="primary"):
    doc_records = []
    rejected_docs = []
    doc_id_counter = 0

    # Fetch live docs (network I/O, no disk needed)
    live_ingested: list[dict] = []
    if has_live:
        with st.spinner("Fetching live data..."):
            live_ingested = fetch_live_docs(
                ticker=live_ticker.strip().upper(),
                company_name=live_company.strip(),
                use_yahoo=use_yahoo,
                use_newsapi=use_newsapi,
                newsapi_key=newsapi_key_input.strip() or None,
                max_per_source=max_live_articles,
            )
        if live_ingested:
            st.info(f"Fetched **{len(live_ingested)}** live article(s). Running them through the pipeline...")
        else:
            st.warning("Live data fetch returned no articles. Check your ticker / API key.")

    total_docs = len(live_ingested) + len(uploaded_files)
    progress = st.progress(0)
    status = st.status("Starting pipeline...", expanded=True)

    with tempfile.TemporaryDirectory() as tmpdir:

        sent_model = load_sentence_model()

        # Pass 0: Live docs (already ingested — skip ingest_file)
        for i, ingested in enumerate(live_ingested):
            status.write(f"🌐 Live: **{ingested['filename']}**")
            progress.progress(i / max(total_docs, 1))
            record = _process_doc(ingested, status, rejected_docs, sentence_model=sent_model)
            if record:
                doc_id_counter += 1
                record["doc_id"] = doc_id_counter
                doc_records.append(record)

        # Pass 1: Uploaded files (ingest from disk, then process)
        for i, uploaded_file in enumerate(uploaded_files):
            status.write(f"📄 Processing: **{uploaded_file.name}**")
            progress.progress((len(live_ingested) + i) / max(total_docs, 1))

            tmp_path = os.path.join(tmpdir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())

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

            record = _process_doc(ingested, status, rejected_docs, sentence_model=sent_model)
            if record:
                doc_id_counter += 1
                record["doc_id"] = doc_id_counter
                doc_records.append(record)

        # FinBERT inference
        status.write("🤖 Running FinBERT sentiment analysis...")
        all_chunks_flat = []
        for doc in doc_records:
            all_chunks_flat.extend(doc["chunks"])

        if not all_chunks_flat:
            st.error("No usable chunks found. All documents were rejected or empty.")
            st.stop()

        finbert_progress = st.progress(0, text="FinBERT inference...")

        def update_finbert_progress(done: int, total: int):
            finbert_progress.progress(done / total, text=f"FinBERT: {done}/{total} chunks")

        all_sentiment_results = run_finbert(
            all_chunks_flat,
            batch_size=finbert_batch_size,
            progress_callback=update_finbert_progress,
        )
        finbert_progress.progress(1.0, text="FinBERT complete")

        # Aggregation & export
        status.write("📊 Computing metrics...")
        full_df = build_results_dataframe(doc_records, all_chunks_flat, all_sentiment_results)
        summary_df = compute_document_summary(full_df)
        portfolio = compute_portfolio_metrics(summary_df)

        output_path = os.path.join(tmpdir, "finsentinel_results.xlsx")
        export_to_excel(full_df, summary_df, rejected_docs, output_path)
        with open(output_path, "rb") as f:
            excel_bytes = f.read()

        progress.progress(1.0)
        status.update(label="✅ Analysis complete!", state="complete")

    # ---------------------------------------------------------------------------
    # Results UI
    # ---------------------------------------------------------------------------
    st.divider()
    st.header("Results")

    # Apply confidence filter
    filtered_df = full_df[full_df["sentiment_confidence"] >= confidence_threshold].copy()
    n_filtered_out = len(full_df) - len(filtered_df)

    # --- Source category mapping (document_type + filename prefix → user label) ---
    _DOC_TYPE_MAP = {
        "equity_research": "Equity Research",
        "10K": "SEC 10-Q/K",
        "10Q": "SEC 10-Q/K",
        "MDA": "MD&A",
        "press_release": "Press Release",
        "earnings_transcript": "Earnings Transcript",
        "news": "News",
        "filing": "Filing",
    }

    def _source_category(row):
        if row["file_type"] == "live_news":
            fname = row["filename"].removeprefix("[headline-only] ")
            return "News API" if fname.startswith("newsapi_") else "Yahoo Finance"
        return _DOC_TYPE_MAP.get(row["document_type"], "Other")

    filtered_df["source_category"] = filtered_df.apply(_source_category, axis=1)

    # --- Neutral reclassification ---
    # neutral >= 0.8 → stays neutral (weight=0, contributes nothing to CWDS)
    # neutral < 0.8  → reclassify as pos/neg with weight = (1 - neutral_score)
    # original pos/neg chunks → keep label, weight = sentiment_confidence
    NEUTRAL_THRESHOLD = 0.8

    def _adjust(row):
        if row["neutral"] >= NEUTRAL_THRESHOLD:
            return pd.Series({"adjusted_label": "neutral", "adjusted_weight": 0.0})
        if row["sentiment_label"] != "neutral":
            return pd.Series({"adjusted_label": row["sentiment_label"], "adjusted_weight": row["sentiment_confidence"]})
        label = "positive" if row["positive"] > row["negative"] else "negative"
        return pd.Series({"adjusted_label": label, "adjusted_weight": 1.0 - row["neutral"]})

    filtered_df[["adjusted_label", "adjusted_weight"]] = filtered_df.apply(_adjust, axis=1)

    # --- CWDS per source category ---
    # CWDS = Σ(direction_i × weight_i) / Σ(weight_i)  where direction = positive - negative
    def _cwds_agg(grp):
        direction = grp["positive"] - grp["negative"]
        weights = grp["adjusted_weight"]
        total_w = weights.sum()
        score = float((direction * weights).sum() / total_w) if total_w > 0 else 0.0
        return pd.Series({
            "cwds": round(score, 4),
            "chunk_count": len(grp),
            "pct_positive": round((grp["adjusted_label"] == "positive").mean() * 100, 1),
            "pct_negative": round((grp["adjusted_label"] == "negative").mean() * 100, 1),
            "pct_neutral":  round((grp["adjusted_label"] == "neutral").mean() * 100, 1),
            "avg_weight":   round(weights.mean(), 3),
        })

    cwds_df = filtered_df.groupby("source_category", group_keys=False).apply(_cwds_agg).reset_index()
    cwds_df = cwds_df.sort_values("cwds", ascending=False)

    total_w = filtered_df["adjusted_weight"].sum()
    overall_cwds = float(
        ((filtered_df["positive"] - filtered_df["negative"]) * filtered_df["adjusted_weight"]).sum() / total_w
    ) if total_w > 0 else 0.0

    # --- Metric cards ---
    col1, col2, col3, col4, col5 = st.columns(5)
    cwds_label = "Bullish" if overall_cwds > 0.05 else ("Bearish" if overall_cwds < -0.05 else "Neutral")
    col1.metric("CWDS (Overall)", f"{overall_cwds:.3f}", delta=cwds_label,
                help="Confidence-Weighted Directional Score: Σ(direction × weight) / Σ(weight)")
    col2.metric("Docs Analyzed", portfolio.get("total_docs_analyzed", 0))
    col3.metric("Chunks Analyzed", portfolio.get("total_chunks_analyzed", 0))
    signal_pct = (filtered_df["adjusted_label"] != "neutral").mean() * 100 if len(filtered_df) > 0 else 0.0
    col4.metric("Signal Strength", f"{signal_pct:.1f}%",
                help="% of chunks with directional signal after neutral reclassification")
    col5.metric("Docs Rejected", len(rejected_docs))

    if n_filtered_out > 0:
        st.caption(f"ℹ️ {n_filtered_out} low-confidence chunks hidden (confidence < {confidence_threshold:.2f}).")
    if portfolio.get("most_bullish_doc") or portfolio.get("most_bearish_doc"):
        st.caption(
            f"Most bullish: **{portfolio.get('most_bullish_doc', 'N/A')}** | "
            f"Most bearish: **{portfolio.get('most_bearish_doc', 'N/A')}**"
        )

    # --- Net sentiment per document ---
    st.subheader("Net Sentiment by Document")
    summary_df["source_type"] = summary_df["file_type"].apply(
        lambda ft: "Live" if ft == "live_news" else "Uploaded"
    )
    fig_bar = px.bar(
        summary_df,
        x="filename",
        y="mean_net_sentiment",
        color="mean_net_sentiment",
        color_continuous_scale=["#d62728", "#aaaaaa", "#2ca02c"],
        color_continuous_midpoint=0,
        labels={"mean_net_sentiment": "Net Sentiment (Pos − Neg)", "filename": "Document"},
        title="Net Sentiment Score per Document",
        hover_data=["document_type", "source_type", "chunk_count", "pct_positive", "pct_negative"],
    )
    fig_bar.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Pie (adjusted labels) + Stacked bar by source category ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Adjusted Sentiment Distribution")
        label_counts = filtered_df["adjusted_label"].value_counts()
        fig_pie = px.pie(
            values=label_counts.values,
            names=label_counts.index,
            color=label_counts.index,
            color_discrete_map={"positive": "#2ca02c", "negative": "#d62728", "neutral": "#aaaaaa"},
            title="After neutral reclassification (threshold=0.8)",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Sentiment by Source")
        src_counts = (
            filtered_df.groupby(["source_category", "adjusted_label"])
            .size()
            .reset_index(name="count")
        )
        src_totals = src_counts.groupby("source_category")["count"].transform("sum")
        src_counts["pct"] = src_counts["count"] / src_totals * 100
        fig_source = px.bar(
            src_counts,
            x="source_category",
            y="pct",
            color="adjusted_label",
            color_discrete_map={"positive": "#2ca02c", "negative": "#d62728", "neutral": "#aaaaaa"},
            labels={"pct": "% of chunks", "source_category": "Source", "adjusted_label": "Sentiment"},
            title="Adjusted Sentiment % by Source",
            barmode="stack",
            text_auto=".1f",
        )
        fig_source.update_layout(yaxis_title="% of chunks", legend_title="Sentiment",
                                  xaxis_tickangle=-20)
        st.plotly_chart(fig_source, use_container_width=True)

    # --- CWDS horizontal bar chart ---
    st.subheader("Confidence-Weighted Directional Score (CWDS) by Source")
    st.caption("CWDS = Σ(direction × weight) / Σ(weight) — neutrals (≥0.8) contribute zero; weak neutrals reclassified with weight = 1 − neutral_score")
    fig_cwds = px.bar(
        cwds_df,
        x="cwds",
        y="source_category",
        orientation="h",
        color="cwds",
        color_continuous_scale=["#d62728", "#aaaaaa", "#2ca02c"],
        color_continuous_midpoint=0,
        labels={"cwds": "CWDS Score", "source_category": "Source"},
        hover_data={"chunk_count": True, "pct_positive": True, "pct_negative": True, "pct_neutral": True, "avg_weight": True},
        text=cwds_df["cwds"].apply(lambda v: f"{v:+.3f}"),
    )
    fig_cwds.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
    fig_cwds.update_traces(textposition="outside")
    st.plotly_chart(fig_cwds, use_container_width=True)

    st.dataframe(
        cwds_df.rename(columns={
            "source_category": "Source", "cwds": "CWDS", "chunk_count": "Chunks",
            "pct_positive": "% Positive", "pct_negative": "% Negative",
            "pct_neutral": "% Neutral", "avg_weight": "Avg Weight",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # --- Document summary ---
    st.subheader("Document Summary")
    st.dataframe(
        summary_df[[
            "filename", "document_type", "source_type", "chunk_count",
            "mean_net_sentiment", "std_net_sentiment",
            "pct_positive", "pct_negative", "pct_neutral",
        ]].round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Full Results (All Chunks)")
    st.caption(f"Showing {len(filtered_df):,} chunks with confidence ≥ {confidence_threshold:.2f}")
    st.dataframe(
        filtered_df[[
            "filename", "document_type", "source_category", "section",
            "sentiment_label", "adjusted_label", "adjusted_weight",
            "positive", "negative", "neutral", "net_sentiment", "token_count",
        ]].round(3),
        use_container_width=True,
        hide_index=True,
    )

    if show_rejected and rejected_docs:
        with st.expander(f"⚠️ Rejected Documents ({len(rejected_docs)})"):
            st.dataframe(pd.DataFrame(rejected_docs), use_container_width=True, hide_index=True)

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
        "📥 Download Summary (CSV)",
        data=summary_df.to_csv(index=False),
        file_name="finsentinel_summary.csv",
        mime="text/csv",
    )
