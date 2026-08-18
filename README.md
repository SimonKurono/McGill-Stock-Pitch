# McGill Stock Pitch: Skeena Real Options Valuation

[![SKE Deck](https://github.com/user-attachments/assets/1413f936-5f9b-4020-b0d4-75415e3f088a)](https://drive.google.com/file/d/1YZKUVQItDEmYmFgGo_YtcLDSRz-3QrAh/view?usp=sharing)

**Link to deck**: [`SKE_Pitch_Deck.pdf`](https://drive.google.com/file/d/1YZKUVQItDEmYmFgGo_YtcLDSRz-3QrAh/view?usp=sharing) (also see [`SKE_DEMO_DECK.pdf`](./SKE_DEMO_DECK.pdf))

**Link to Model**: https://docs.google.com/spreadsheets/d/1MBIdMbClGtRW02CdPVe5KgcLT_cLjyU8/edit?usp=sharing&ouid=117316930616149647827&rtpof=true&sd=true and [`SKE_MODEL.xlsx`](./SKE_MODEL.xlsx)

A compact valuation project that estimates the development flexibility of Skeena Gold & Silver (TSX: `SKE`) using a Black-Scholes-style real options framework.

## Overview

Traditional NAV can understate pre-production mining value by treating development as fixed. This project models management flexibility as a call option on project value:

`Equity Value = Base NAV + Real Option Value`

The analysis is implemented in [`black-scholes.ipynb`](./black-scholes.ipynb) and includes base-case valuation plus sensitivity analysis.

## What the Notebook Does

- Defines project assumptions (`S`, `K`, `T`, `r`, `sigma`, shares outstanding, base gold price)
- Computes total option value with a Black-Scholes approximation
- Converts option value to per-share uplift and adjusted NAV/share
- Produces:
  - NAV vs option value bar chart
  - Option decomposition vs gold price (option, intrinsic, time value)
  - Sensitivity heatmaps for:
    - Gold price x capex
    - Time x risk-free rate
- Prints a preview output table for scenario analysis

## Repository Structure

- [`black-scholes.ipynb`](./black-scholes.ipynb): main analysis
- [`requirements.txt`](./requirements.txt): Python dependencies
- [`nlp/finsentinel/`](./nlp/finsentinel): a separate project, FinSentinel — see below

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook black-scholes.ipynb
```

## Current Base-Case Output (from notebook run)

- Total option value: `4,559,294,619.42`
- Option value per share: `3.7331`
- Adjusted NAV per share: `56.3631`

## Notes

- This is an approximation framework for pitch/investment discussion, not a production valuation engine.
- Results are highly sensitive to volatility, capex, timing, and commodity-price assumptions.
- Update Section 2 inputs in the notebook before re-running scenarios.

## FinSentinel

This repo also includes [`nlp/finsentinel/`](./nlp/finsentinel), a Streamlit app that classifies and scores the sentiment of financial documents (10-K/Q filings, MD&A, equity research, news) using Claude for document understanding and FinBERT for sentiment scoring. It was built to support this same SKE research process. See [`nlp/finsentinel/README.md`](./nlp/finsentinel/README.md) for setup and usage.
