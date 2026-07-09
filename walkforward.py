"""Walk-forward tuning harness — tune on old data, trade the next year,
never let the optimizer see the future.

Spec:
  - Build/tune on in-sample (IS) windows only; walk forward through time
  - Selection filters per window, applied IN-SAMPLE:
      * Max drawdown better than -35%
      * Then highest Sharpe wins (higher = better)
  - Overfit test per window: NOT overfit iff  IS_Sharpe <= 1.3 * OOS_Sharpe + 0.5
    (in-sample performance may not exceed what out-of-sample supports,
    allowing 30% degradation + 0.5 noise)
  - IS/OOS plot: stitched walk-forward equity vs buy-and-hold, plus
    per-window IS-vs-OOS Sharpe bars

Strategy family under test: TSMOM/vol-target variants on SPY (a widely
promoted retail strategy family, tuned honestly). Grid = 216 variants:
  momentum lookback {63,126,189,252} x vol target {10,15,20%} x
  vol window {30,60,90} x base leverage {1.0,1.5,2.0} x dd-protection {on,off}

Walk-forward: 5y training window, select, trade 1y OOS, roll annually.
Costs: 0.1% per unit of traded notional. Known simplification: the dd
overlay runs on each variant's full-history equity (state crosses window
boundaries) — conservative and consistent across variants.

Requires: numpy, pandas, matplotlib, statsmodels, yfinance.
"""

from __future__ import annotations

import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from validation import block_bootstrap_sharpe, newey_west_tstat

ANN = 252
COST = 0.001
OUT = Path(__file__).resolve().parent / "output"
GRID = list(itertools.product(
    [63, 126, 189, 252],          # momentum lookback
    [0.10, 0.15, 0.20],           # vol target
    [30, 60, 90],                 # vol window
    [1.0, 1.5, 2.0],              # base leverage
    [True, False],                # dd protection
))

BLUE, INK, MUTED, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb"
GRID_C = "#e5e4e0"


def variant_returns(spy: pd.Series, ret: pd.Series, mom_lb: int, vt: float,
                    vw: int, lev: float, dd_on: bool) -> pd.Series:
    mom = spy / spy.shift(mom_lb) - 1
    vol = ret.rolling(vw).std() * np.sqrt(ANN)
    sizing = ((vt / vol) * lev).clip(0.5, 3.0)
    month = spy.index.to_period("M")
    is_rebal = pd.Series(month != np.roll(month, 1), index=spy.index)
    sig = ((mom > 0).astype(float) * sizing).shift(1)
    pos = sig.where(is_rebal).ffill().fillna(0.0).values

    # drawdown protection must be EDGE-triggered with an explicit release:
    # a level-triggered "flat at -25%" rule freezes forever once flat
    # (equity can't move, so the level never un-breaches).
    r_out = np.zeros(len(spy))
    equity, peak, cooldown, prev_p = 1.0, 1.0, 0, 0.0
    rv = ret.values
    for i in range(len(spy)):
        p = pos[i]
        if dd_on:
            dd = equity / peak - 1
            if cooldown > 0:
                p = 0.0
                cooldown -= 1
                if cooldown == 0:
                    peak = equity
            elif dd <= -0.25:
                p, cooldown = 0.0, 20
            elif dd <= -0.15:
                p *= 0.5
        r = p * rv[i] - abs(p - prev_p) * COST
        equity *= (1 + r)
        peak = max(peak, equity)
        prev_p = p
        r_out[i] = r
    return pd.Series(r_out, index=spy.index)


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(ANN)) if len(r) > 20 and r.std() > 0 else 0.0


def maxdd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def main() -> None:
    import yfinance as yf

    OUT.mkdir(exist_ok=True)
    spy = yf.download("SPY", start="2004-01-01", auto_adjust=True,
                      progress=False)["Close"].squeeze()
    ret = spy.pct_change().fillna(0)

    print(f"computing {len(GRID)} variants...")
    variants = {g: variant_returns(spy, ret, *g) for g in GRID}

    years = sorted(set(spy.index.year))
    oos_years = [y for y in years if y >= years[0] + 6 and y <= years[-1]]
    windows, oos_parts = [], []
    for y in oos_years:
        is_mask = (spy.index.year >= y - 5) & (spy.index.year < y)
        oos_mask = spy.index.year == y
        best, best_s = None, -np.inf
        for g, r in variants.items():
            r_is = r[is_mask]
            if maxdd(r_is) <= -0.35:
                continue
            s = sharpe(r_is)
            if s > best_s:
                best, best_s = g, s
        if best is None:
            continue
        r_oos = variants[best][oos_mask]
        s_oos = sharpe(r_oos)
        overfit = best_s > 1.3 * s_oos + 0.5
        windows.append({"year": int(y), "params": best,
                        "is_sharpe": round(best_s, 2),
                        "oos_sharpe": round(s_oos, 2),
                        "overfit_flag": bool(overfit)})
        oos_parts.append(r_oos)

    wf = pd.concat(oos_parts)
    bh = ret.loc[wf.index]
    excess = wf - bh
    t_ex = newey_west_tstat(excess)
    p_ex, _ = block_bootstrap_sharpe(excess)
    n_of = sum(w["overfit_flag"] for w in windows)

    def stats(r):
        eq = (1 + r).cumprod()
        yrs = len(r) / ANN
        return {"total_x": round(float(eq.iloc[-1]), 2),
                "cagr": round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4),
                "sharpe": round(sharpe(r), 2), "maxdd": round(maxdd(r), 3)}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[3, 2],
                                   facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID_C)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(True, color=GRID_C, linewidth=0.6, alpha=0.7)

    eq_wf = (1 + wf).cumprod()
    eq_bh = (1 + bh).cumprod()
    for i, w in enumerate(windows):
        if i % 2 == 0:
            y = w["year"]
            ax1.axvspan(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31"),
                        color=GRID_C, alpha=0.35, linewidth=0)
    ax1.plot(eq_wf.index, eq_wf.values, color=BLUE, linewidth=2)
    ax1.plot(eq_bh.index, eq_bh.values, color=MUTED, linewidth=2)
    ax1.set_yscale("log")
    ax1.text(eq_wf.index[-1], eq_wf.iloc[-1], "  walk-forward (OOS only)",
             color=BLUE, fontsize=10, va="center", fontweight="bold")
    ax1.text(eq_bh.index[-1], eq_bh.iloc[-1], "  buy & hold",
             color=MUTED, fontsize=10, va="center")
    ax1.set_title("Walk-forward TSMOM on SPY — every point traded out-of-sample "
                  "(tuned only on the prior 5 years)",
                  color=INK, fontsize=12, loc="left", pad=12)
    ax1.set_xlim(eq_wf.index[0], eq_wf.index[-1] + pd.Timedelta(days=800))

    xs = np.arange(len(windows))
    is_v = [w["is_sharpe"] for w in windows]
    oos_v = [w["oos_sharpe"] for w in windows]
    ax2.bar(xs - 0.2, is_v, width=0.38, color=MUTED, alpha=0.55,
            label="IS Sharpe (tuning window)")
    ax2.bar(xs + 0.2, oos_v, width=0.38, color=BLUE,
            label="OOS Sharpe (traded year)")
    for i, w in enumerate(windows):
        if w["overfit_flag"]:
            ax2.text(xs[i], max(is_v[i], oos_v[i]) + 0.08, "×", color="#e34948",
                     ha="center", fontsize=11, fontweight="bold")
    ax2.axhline(0, color=MUTED, linewidth=0.8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(w["year"]) for w in windows], rotation=0, fontsize=8)
    ax2.set_title("Per-window: what tuning promised (gray) vs what the next "
                  "year paid (blue) — × = overfit flag (IS > 1.3×OOS + 0.5)",
                  color=INK, fontsize=11, loc="left", pad=10)
    ax2.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper left")

    plt.tight_layout()
    png = OUT / f"walkforward_{date.today().isoformat()}.png"
    fig.savefig(png, dpi=150, facecolor=SURFACE, bbox_inches="tight")

    lines = [f"# Walk-forward TSMOM study — {date.today().isoformat()}",
             f"\n{len(GRID)} variants | 5y tune → 1y trade | OOS years "
             f"{windows[0]['year']}–{windows[-1]['year']}\n",
             f"- Walk-forward OOS: {json.dumps(stats(wf))}",
             f"- Buy & hold:       {json.dumps(stats(bh))}",
             f"- Excess: NW t = {t_ex:.2f}, bootstrap P(Sharpe<=0) = {p_ex:.3f}",
             f"- Overfit-flagged windows: {n_of}/{len(windows)} "
             f"(rule: IS > 1.3*OOS + 0.5)",
             f"- Plot: {png.name}", "\nPer-window detail:"]
    for w in windows:
        lines.append(f"  - {w['year']}: IS {w['is_sharpe']:+.2f} -> OOS "
                     f"{w['oos_sharpe']:+.2f} {'OVERFIT' if w['overfit_flag'] else 'ok'} "
                     f"params={w['params']}")
    out = OUT / f"walkforward_{date.today().isoformat()}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:10]))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
