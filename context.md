# `context.md` (Comprehensive + Rigorous)

Replace your current `context.md` with this:

````md
# Skeena Real Options Valuation — Full Context

---

# 1. Objective

Develop a **rigorous, defensible real options valuation framework** for Skeena Gold & Silver (Eskay Creek project) that complements a traditional NAV model.

The goal is to demonstrate:

> Static NAV undervalues pre-production mining assets because it ignores managerial flexibility under commodity price uncertainty.

Final valuation framework:

```math
\text{Equity Value} = \text{Base NAV} + \text{Real Option Value}
````

This project must:

* be **quantitatively credible**
* be **implementable in 1–2 days**
* be **defensible under questioning**
* produce **clean, presentation-ready outputs**

---

# 2. Economic Framework

## 2.1 Core Insight

Skeena is not a passive asset.

It is a **decision under uncertainty**.

Management holds the right (but not obligation) to develop the project depending on economic conditions.

This creates an **option-like payoff structure**.

---

## 2.2 Option Payoff

The project behaves like:

```math
\max(V_{project} - K, 0)
```

Where:

* (V_{project}) = present value of project cash flows
* (K) = required development capex

Interpretation:

* If (V_{project} > K): build → capture value
* If (V_{project} < K): delay or avoid → limit losses

---

## 2.3 What the Option Is On

**Critical clarification:**

The option is NOT directly on gold.

It is:

> A call option on project value, where project value is driven by gold prices.

Formally:

```math
V_{project} = f(\text{Gold Price}, \text{Costs}, \text{Reserves})
```

Gold is the **primary driver of uncertainty**, but not the underlying asset itself.

---

## 2.4 Why This Matters

Traditional NAV assumes:

* fixed commodity prices
* fixed development timing
* fixed operating plan

This ignores:

* timing flexibility
* nonlinear exposure to gold prices
* managerial decision-making

Result:

```math
\text{NAV understates intrinsic value}
```

---

# 3. Modeling Approach
## 3.1 Chosen Method

Use a **Black-Scholes-style approximation** to estimate the value of development flexibility.

This is NOT a literal claim that the project is a traded option.

It is an **economic approximation under uncertainty**.

---

## 3.2 Variable Mapping

| Black-Scholes Variable | Mining Interpretation                                    |
| ---------------------- | -------------------------------------------------------- |
| (S)                    | Project value (proxy from NAV model)                     |
| (K)                    | Remaining capex required to develop                      |
| (T)                    | Time until development decision / production             |
| (σ)                    | Volatility of project value (proxied by gold volatility) |
| (r)                    | Risk-free rate (Treasury yield)                          |

---

## 3.3 Black-Scholes Formula

```math
C = S N(d_1) - K e^{-rT} N(d_2)
```

Where:

```math
d_1 = \frac{\ln(S/K) + (r + \frac{σ^2}{2})T}{σ\sqrt{T}}
```

```math
d_2 = d_1 - σ\sqrt{T}
```

Output:

* option value in dollars
* option value per share
* adjusted equity value

---

# 4. Key Assumptions

## 4.1 Project Value Proxy (S)

We approximate (S) using:

* base-case NAV or project value from the existing model

This is an approximation because:

* true project value is stochastic
* NAV is a deterministic estimate

---

## 4.2 Strike Price (K)

Use:

* remaining capex required to construct the mine

This represents the cost to "exercise" the option.

---

## 4.3 Time to Maturity (T)

Use:

* time until commercial production
* or time until development becomes irreversible

Expected range:

* ~1.5–2.5 years

---

## 4.4 Volatility (σ)

We assume:

```math
σ_{project} \approx σ_{gold}
```

Justification:

* project value is primarily driven by gold prices
* especially for a pre-production asset

Compute:

```math
σ = \text{std}(\text{daily returns}) × \sqrt{252}
```

Use:

* 1-year and 3-year realized volatility
* choose conservative estimate

---

## 4.5 Risk-Free Rate (r)

Use:

* 2-year US Treasury yield

Reason:

* matches time horizon (T)

---

# 5. Model Outputs

## 5.1 Core Outputs

* Option value (total)
* Option value per share
* Adjusted value per share

---

## 5.2 Supporting Outputs

* Sensitivity table (σ vs T)
* Waterfall bridge (NAV → +Option → Adjusted value)
* Assumptions table

---

# 6. Sensitivity Analysis

## 6.1 Required Sensitivities

Vary:

* volatility (σ)
* time (T)

Example grid:

| σ \ T | 1.5 yrs | 2.0 yrs | 2.5 yrs |
| ----- | ------- | ------- | ------- |
| 15%   |         |         |         |
| 20%   |         |         |         |
| 25%   |         |         |         |

---

## 6.2 Expected Behavior

The model must satisfy:

* (↑σ → ↑Option Value)
* (↑T → ↑Option Value)
* (↑S → ↑Option Value)
* (↑K → ↓Option Value)

These must be validated in tests.

---

# 7. Implementation Plan

## 7.1 Modules

### `data_loader.py`

* load gold prices
* load project inputs
* clean and structure data

### `volatility.py`

* compute returns
* compute annualized volatility
* compare lookback windows

### `black_scholes.py`

* compute d1, d2
* compute call price
* validate monotonicity

### `valuation.py`

* compute per-share option value
* combine with NAV
* output summary table

### `sensitivity.py`

* build 2D grids
* return DataFrame

### `charts.py`

* waterfall chart
* heatmap

---

## 7.2 Coding Requirements

* use type hints
* include docstrings
* modular functions
* no hardcoded constants
* explicit assumptions
* reproducible outputs

---

# 8. Validation

## 8.1 Numerical Validation

Check:

* monotonicity
* reasonable magnitude
* stability across inputs

---

## 8.2 Economic Validation

Ensure:

* option value is incremental (not dominant)
* results align with intuition
* values are plausible relative to market cap

---

# 9. Limitations

Explicitly acknowledge:

1. Black-Scholes assumes continuous trading (not true)
2. Project value is not directly observable
3. Volatility proxy is imperfect
4. No explicit modeling of operational constraints
5. Ignores multi-stage decision structure

Conclusion:

> This is an approximation designed to capture economic intuition, not an exact valuation.

---

# 10. Investment Interpretation

## 10.1 Core Conclusion

Static NAV underestimates Skeena’s value because it ignores:

* development timing flexibility
* convex exposure to gold prices
* strategic optionality

---

## 10.2 Actionable Takeaway

> Skeena should be viewed as a pre-production asset with embedded optionality, providing leveraged exposure to gold with partially limited downside, supporting valuation above static NAV.

---

# 11. Communication Strategy

## 11.1 Short Explanation

“Skeena behaves like a call option on its project value. Because management can delay development under uncertainty, the project has embedded option value that static NAV does not capture.”

---

## 11.2 Defense Points

### Why Black-Scholes?

Used as a tractable approximation for valuing flexibility.

### Why gold volatility?

Project value is primarily driven by gold prices.

### Why not in NAV?

NAV assumes fixed decisions, not flexible ones.

---

# 12. Extensions (Optional)

If time permits:

* exploration option (probability-weighted NAV uplift)
* takeout option (premium × probability)
* Monte Carlo simulation of gold paths

---


