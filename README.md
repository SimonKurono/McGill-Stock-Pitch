# McGill Stock Pitch: Skeena Real Options Valuation
**Link to deck**: https://drive.google.com/file/d/1YZKUVQItDEmYmFgGo_YtcLDSRz-3QrAh/view?usp=sharing and [`SKE_Pitch_Deck.pdf`](./SKE_Pitch_Deck.pdf)

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
- [`context.md`](./context.md): modeling context and economic framing
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
