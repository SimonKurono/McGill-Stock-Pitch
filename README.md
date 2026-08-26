# McGill Stock Pitch: TSX:SKE
This repository contains the following 4 documents:
1. ['Slide Deck'](./SKE_DEMO_DECK.pdf)
2. ['NAV Model'](./SKE_MODEL.xlsx)
3. ['Real Option Valuation Notebook'](./black-scholes.ipynb)
4. ['FinBERT NLP Pipeline'](./nlp/finsentinel) (Watch demo [here](https://drive.google.com/file/d/1QNsvRXIuYhcDe121b6JF65_DT76QKQJU/view))

[![SKE Deck](https://github.com/user-attachments/assets/1413f936-5f9b-4020-b0d4-75415e3f088a)]([https://drive.google.com/file/d/1YZKUVQItDEmYmFgGo_YtcLDSRz-3QrAh/view?usp=sharing](https://drive.google.com/drive/folders/1ut7T71-La0klRtzPvNqBcZzRDnf0VZJa))

This project is a compact valuation project that estimates the development flexibility of Skeena Gold & Silver (TSX: `SKE`) using a Black-Scholes-style real options framework and provides supplementary analysis regarding overall sentiment toward SKE. Using 1300+ datapoints from MD&A, SEC filings, equity research reports, and news sources, price-relevant sentiment features were constructed as an alternative data analysis source. MiniLM semantic clustering was applied for credibility/neutrality filtering to extract 'pure' sentiment signals.

## Real Options Valuation Overview

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

## NLP Pipeline: FinSentinel

This repo also includes [`nlp/finsentinel/`](./nlp/finsentinel), a Streamlit app that classifies and scores the sentiment of financial documents (10-K/Q filings, MD&A, equity research, news) using Claude for document understanding and FinBERT for sentiment scoring. It was built to support this same SKE research process. See [`nlp/finsentinel/README.md`](./nlp/finsentinel/README.md) for setup and usage.


