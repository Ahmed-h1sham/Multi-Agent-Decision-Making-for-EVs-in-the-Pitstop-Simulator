"""Validate the station/queue core against Erlang C on a single M/M/c station.

Uses the same Station class (start_ready / finish) as the EV simulation, driven with
Poisson arrivals and exponential service times. This reproduces the curves in the
background chapter (Figure "Mean queue waiting against utilisation").
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from scipy import stats

from .config import StationSpec
from .station import FCFS, Station, Waiting
from .stats import erlang_c_wq


def simulate_mmc(c: int, rho: float, n: int = 200_000, seed: int = 7,
                 warmup: float = 0.1, batches: int = 20) -> dict:
    rng = np.random.default_rng(seed)
    lam = c * rho                                   # service rate mu = 1 per charger
    arrivals = np.cumsum(rng.exponential(1.0 / lam, n))
    services = rng.exponential(1.0, n)
    st = Station(StationSpec(0, "M/M/c", 0, 0, "dc_fast", c), FCFS)
    start = np.full(n, np.nan)
    finishes = []                                   # heap of (t, charger idx)
    i = 0
    while i < n or finishes:
        if finishes and (i >= n or finishes[0][0] <= arrivals[i]):
            t, idx = heapq.heappop(finishes)
            st.finish(idx, t)
        else:
            t = arrivals[i]
            st.join(Waiting((1, t, i), i, t, services[i]))
            i += 1
        for w, idx, t_fin in st.start_ready(t):
            start[w.rid] = t
            heapq.heappush(finishes, (t_fin, idx))
    wq = (start - arrivals)[int(warmup * n):]
    means = np.array([b.mean() for b in np.array_split(wq, batches)])
    half = stats.t.ppf(0.975, batches - 1) * means.std(ddof=1) / math.sqrt(batches)
    return {"c": c, "rho": rho, "analytic_wq": erlang_c_wq(c, rho),
            "sim_wq": float(wq.mean()), "ci_low": float(wq.mean() - half),
            "ci_high": float(wq.mean() + half), "max_busy": st.max_busy,
            "violations": len(st.violations)}


def erlang_table(cs=(1, 2, 4), rhos=(0.5, 0.7, 0.9), n: int = 200_000) -> list:
    return [simulate_mmc(c, r, n=n, seed=1000 * c + int(r * 100)) for c in cs for r in rhos]
