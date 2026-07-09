"""Stage 4 — statistical validation. The rigor lives here.

Three tests, all must pass:
  1. Newey-West t-stat on the mean daily return (HAC-robust) >= threshold.
  2. Moving-block bootstrap (10k draws) — P(Sharpe <= 0) must be small.
  3. In-sample (first 70%) vs out-of-sample (last 30%) Sharpe degradation
     <= 30%. Worse than that is the signature of overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

ANN = 252


@dataclass
class ValidationResult:
    passed: bool
    nw_tstat: float
    bootstrap_p: float
    sharpe_quantiles: dict
    sharpe_is: float
    sharpe_oos: float
    degradation: float
    reasons: list[str]

    def summary(self) -> dict:
        return {
            "passed": self.passed,
            "nw_tstat": round(self.nw_tstat, 2),
            "bootstrap_p_sharpe_le_0": round(self.bootstrap_p, 4),
            # distributional view, not a point estimate
            "bootstrap_sharpe_p5_p50_p95": self.sharpe_quantiles,
            "sharpe_is": round(self.sharpe_is, 3),
            "sharpe_oos": round(self.sharpe_oos, 3),
            "is_oos_degradation": round(self.degradation, 3),
            "reasons": self.reasons,
        }


def newey_west_tstat(returns: pd.Series) -> float:
    """t-stat of the mean with HAC standard errors (Newey-West auto lag)."""
    y = returns.values
    lags = int(np.floor(4 * (len(y) / 100) ** (2 / 9)))
    model = sm.OLS(y, np.ones(len(y))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(model.tvalues[0])


def block_bootstrap_sharpe(returns: pd.Series, n_iter: int = 10_000,
                           block: int = 21, seed: int = 42) -> tuple[float, np.ndarray]:
    """Moving-block bootstrap of the annualized Sharpe. Returns P(Sharpe<=0)."""
    rng = np.random.default_rng(seed)
    y = returns.values
    n = len(y)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block, size=(n_iter, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_iter, -1)[:, :n]
    samples = y[idx]
    means = samples.mean(axis=1)
    stds = samples.std(axis=1)
    sharpes = np.where(stds > 0, means / stds * np.sqrt(ANN), 0.0)
    return float((sharpes <= 0).mean()), sharpes


def _sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0


def validate(returns: pd.Series, t_threshold: float = 2.5,
             p_threshold: float = 0.05, max_degradation: float = 0.30,
             is_frac: float = 0.7) -> ValidationResult:
    reasons: list[str] = []

    t = newey_west_tstat(returns)
    if t < t_threshold:
        reasons.append(f"Newey-West t={t:.2f} < {t_threshold}")

    p, sharpes = block_bootstrap_sharpe(returns)
    quantiles = {q: round(float(np.percentile(sharpes, q)), 3) for q in (5, 50, 95)}
    if p > p_threshold:
        reasons.append(f"bootstrap P(Sharpe<=0)={p:.3f} > {p_threshold}")

    split = int(len(returns) * is_frac)
    s_is, s_oos = _sharpe(returns.iloc[:split]), _sharpe(returns.iloc[split:])
    degradation = 1 - s_oos / s_is if s_is > 0 else 1.0
    if s_is <= 0:
        reasons.append(f"in-sample Sharpe {s_is:.2f} <= 0")
    elif degradation > max_degradation:
        reasons.append(
            f"IS/OOS degradation {degradation:.0%} > {max_degradation:.0%} "
            f"(IS {s_is:.2f} -> OOS {s_oos:.2f}) — overfitting signature")

    return ValidationResult(
        passed=not reasons, nw_tstat=t, bootstrap_p=p,
        sharpe_quantiles=quantiles,
        sharpe_is=s_is, sharpe_oos=s_oos, degradation=float(degradation),
        reasons=reasons,
    )


def signal_calibration(signal: pd.DataFrame, returns: pd.DataFrame,
                       n_buckets: int = 5) -> dict:
    """Calibration check:
    a real cross-sectional signal should order future returns monotonically,
    not just win top-vs-bottom. Reports daily rank-IC and per-quintile mean
    forward returns. Informational — the critic judges."""
    sig = signal.shift(1)  # information lag, same as the backtester
    common = sig.index.intersection(returns.index)
    sig, rets = sig.loc[common], returns.loc[common]

    # daily rank IC: cross-sectional Spearman via z-scored ranks
    sr = sig.rank(axis=1)
    rr = rets.rank(axis=1)
    sz = sr.sub(sr.mean(axis=1), axis=0).div(sr.std(axis=1), axis=0)
    rz = rr.sub(rr.mean(axis=1), axis=0).div(rr.std(axis=1), axis=0)
    ic = (sz * rz).mean(axis=1).dropna()
    ic_t = float(ic.mean() / ic.std() * np.sqrt(len(ic))) if ic.std() > 0 else 0.0

    # per-quintile mean forward return (bps/day), quintile 1 = lowest signal
    q = sig.rank(axis=1, pct=True)
    buckets = {}
    for b in range(n_buckets):
        mask = (q > b / n_buckets) & (q <= (b + 1) / n_buckets)
        buckets[f"q{b + 1}"] = round(float(rets[mask].stack().mean() * 1e4), 2)
    vals = list(buckets.values())
    monotone_share = float(np.mean([vals[i + 1] >= vals[i]
                                    for i in range(len(vals) - 1)]))

    # Grinold-Kahn Fundamental Law: IR = IC x sqrt(breadth). Breadth here is
    # a deliberate UPPER BOUND (avg names x 12 monthly bets, independence
    # assumed). A backtest Sharpe far above this implied IR claims skill the
    # measured IC cannot support — an overfitting signature the critic checks.
    breadth_names = float(sig.notna().sum(axis=1).mean())
    implied_ir = float(ic.mean()) * np.sqrt(breadth_names * 12)

    # ICIR: consistency of the IC (mean/std) — a modest steady IC beats a
    # flashy unstable one. Signal half-life via daily cross-sectional rank
    # autocorrelation, t1/2 = -ln2/ln(rho): a half-life far below the
    # rebalance period means the strategy pays turnover to chase noise.
    icir = float(ic.mean() / ic.std()) if ic.std() > 0 else 0.0
    sl = sig.shift(1).rank(axis=1)
    lz = sl.sub(sl.mean(axis=1), axis=0).div(sl.std(axis=1), axis=0)
    rho = float((sz * lz).mean(axis=1).dropna().mean())
    if 0 < rho < 1:
        halflife = min(float(-np.log(2) / np.log(rho)), 9999.0)
    else:
        halflife = 9999.0 if rho >= 1 else 0.0

    return {"mean_daily_rank_ic": round(float(ic.mean()), 4),
            "ic_tstat": round(ic_t, 2),
            "icir_daily": round(icir, 4),
            "signal_halflife_days": round(halflife, 1),
            "quintile_fwd_bps_per_day": buckets,
            "monotone_share": round(monotone_share, 2),
            "fundamental_law": {"breadth_names": round(breadth_names),
                                "implied_ir_upper": round(implied_ir, 2)}}
