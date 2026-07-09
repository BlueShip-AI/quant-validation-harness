# quant-validation-harness

The statistical gates behind [The Risk Museum](https://blueship-ai.github.io/the-risk-museum/) —
the validation machinery of the BlueShip-AI research pipeline, open-sourced.
The gates are not the secret; the discipline of never loosening them is.

## What's here

**`validation.py`** — the stage every strategy must survive:

1. **Newey-West t-statistic** on mean daily return (HAC-robust,
   automatic lag selection) — must clear 2.5.
2. **Moving-block bootstrap** of the annualized Sharpe (10,000 draws,
   21-day blocks) — P(Sharpe ≤ 0) must be ≤ 0.05. Reported as a
   distribution (p5/p50/p95), never a point estimate.
3. **IS/OOS degradation** — out-of-sample Sharpe (last 30%) may not
   degrade more than 30% from in-sample. Worse is the signature of
   overfitting.

Plus `signal_calibration()`: daily rank-IC with t-stat, ICIR, per-quintile
monotonicity, signal half-life from rank autocorrelation, and a
Grinold-Kahn Fundamental Law check (IR = IC·√breadth) — a backtest Sharpe
far above the IC-implied upper bound is claiming skill the measured
information cannot support.

**`walkforward.py`** — a complete, runnable walk-forward tuning study:
216 TSMOM/vol-target variants on SPY, tuned on 5-year windows, traded the
next year, rolled annually. Selection filters applied strictly in-sample;
per-window overfit flags (IS > 1.3×OOS + 0.5); output chart shows what
tuning promised vs what the next year paid.

Its finding, reproducible in one command: on public price data,
**the tuning is the overfitting** — the walk-forward stitch underperforms
buy-and-hold (0.47 vs 0.86 Sharpe, 2010–2026).

## Usage

```python
from validation import validate

result = validate(daily_returns)          # a pd.Series of daily returns
print(result.summary())
# {'passed': False, 'nw_tstat': 1.84, 'bootstrap_p_sharpe_le_0': 0.11, ...}
```

```bash
pip install numpy pandas statsmodels matplotlib yfinance
python walkforward.py                     # full study, ~3 minutes
```

## Design principles

- **Deterministic statistics.** In the parent pipeline, LLM agents do
  language work (reading papers, writing critiques); every number comes
  from this code. An LLM never computes a Sharpe ratio.
- **Gates never loosen.** A loosened gate manufactures fake survivors.
  Thresholds are code defaults; changing them is a human decision.
- **Failures are the product.** Everything this harness rejects is
  published, with reasons, at The Risk Museum.

## License

MIT. If it stops one over-fitted backtest from going live, it has paid
for itself.
