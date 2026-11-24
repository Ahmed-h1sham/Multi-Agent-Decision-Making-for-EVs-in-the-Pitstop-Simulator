"""Queueing formulas and the paired comparison.

Paired design (background, "Statistical Comparison"): for scenario j the two policies
give a pair, Delta_j = y_{j,B} - y_{j,A}. The mean difference is reported with a 95 %
confidence interval and a paired t-test. Welch's test is computed only to show what the
old evaluation (independent-samples t-tests, reuse-map issue 8) would have reported.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def erlang_c_wq(c: int, rho: float, mu: float = 1.0) -> float:
    """Mean queue wait W_q of an M/M/c queue (Erlang C), in units of 1/mu when mu = 1."""
    a = c * rho
    tail = a ** c / math.factorial(c) / (1.0 - rho)
    p_wait = tail / (sum(a ** k / math.factorial(k) for k in range(c)) + tail)
    return p_wait / (c * mu * (1.0 - rho))


def paired_summary(y_a, y_b) -> dict:
    y_a, y_b = np.asarray(y_a, float), np.asarray(y_b, float)
    ok = np.isfinite(y_a) & np.isfinite(y_b)
    a, b = y_a[ok], y_b[ok]
    m = int(ok.sum())
    out = {"m": m, "undefined_pairs": int((~ok).sum()),
           "mean_a": float(a.mean()) if m else math.nan,
           "mean_b": float(b.mean()) if m else math.nan}
    if m < 2:
        return out
    diff = b - a
    sd = float(diff.std(ddof=1))
    half = float(stats.t.ppf(0.975, m - 1) * sd / math.sqrt(m))
    p_paired = float(stats.ttest_rel(b, a).pvalue) if sd > 0 else (1.0 if diff.mean() == 0 else 0.0)
    welch = stats.ttest_ind(b, a, equal_var=False)
    va, vb = a.var(ddof=1) / m, b.var(ddof=1) / m
    se_welch = math.sqrt(va + vb)
    df_welch = (va + vb) ** 2 / (va ** 2 / (m - 1) + vb ** 2 / (m - 1)) if se_welch > 0 else m - 1
    out.update({
        "mean_diff": float(diff.mean()),
        "ci_low": float(diff.mean() - half),
        "ci_high": float(diff.mean() + half),
        "p_paired": p_paired,
        "p_welch": float(welch.pvalue) if se_welch > 0 else math.nan,
        "ci_half_paired": half,
        "ci_half_unpaired": float(stats.t.ppf(0.975, df_welch) * se_welch),
        "diffs": [float(x) for x in diff],
    })
    return out
