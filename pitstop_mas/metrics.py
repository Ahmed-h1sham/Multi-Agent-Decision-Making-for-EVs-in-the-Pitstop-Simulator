"""Run-level metrics (background, Section "Evaluation Concepts").

  w_i   = t_start - t_arr                 queue waiting at the station
  d_i   = t_start - t_need                charging-access delay (includes travel)
  mean_w, p_tau over S = requests that started charging
  C     = N_complete / N_req              over all requests activated in the window

A measure whose denominator is empty is undefined (NaN), not zero, and the run is
reported rather than dropped. Failed and unfinished requests are counted separately.

With the defer action, time spent deferring is part of d but not of w, so a deferring
policy must be judged on d and C; w alone would make deferral look free.
"""
from __future__ import annotations

import math

import numpy as np

OUTCOMES = ("complete", "abandoned", "stranded", "unfinished_waiting",
            "unfinished_charging", "unfinished_travel", "unfinished_deferred")


TAU_SENSITIVITY = (5, 10, 30)   # extra thresholds reported next to the fixed tau


def run_metrics(sim) -> dict:
    """Metrics over the measured requests: activated in [t_warm, t_warm + t_obs)."""
    cfg = sim.cfg
    recs = [r for r in sim.recs.values() if sim.sc.measured(r.t_need)]
    n_req = len(recs)
    started = [r for r in recs if r.t_start is not None]
    decided = [r for r in recs if r.sid is not None]
    w = np.array([r.w for r in started])
    d = np.array([r.d for r in started])
    err = np.array([r.t_start - r.est_start for r in started])
    nan = math.nan
    counts = {o: sum(r.outcome == o for r in recs) for o in OUTCOMES}
    per_station = [sum(r.sid == st.spec.sid and r.t_arr is not None for r in recs) for st in sim.stations]
    return {
        "level": sim.sc.level,
        "scenario": sim.sc.index,
        "digest": sim.sc.digest(),
        "policy": sim.policy.key,
        "n_req": n_req,
        "n_started": len(started),
        "n_complete": counts["complete"],
        "completion": counts["complete"] / n_req if n_req else nan,
        "mean_w": float(w.mean()) if len(w) else nan,
        "p_tau": float((w > cfg.tau).mean()) if len(w) else nan,
        "mean_d": float(d.mean()) if len(d) else nan,
        "p95_w": float(np.percentile(w, 95)) if len(w) else nan,
        "max_w": float(w.max()) if len(w) else nan,
        "est_error": float(err.mean()) if len(err) else nan,
        "decision_us": float(np.mean([r.decision_ns for r in decided]) / 1000) if decided else nan,
        **{f"p_w{t}": (float((w > t).mean()) if len(w) else nan) for t in TAU_SENSITIVITY},
        "utilisation": sim.busy_minutes / sim.capacity_minutes if sim.capacity_minutes else nan,
        "idle_while_queued": sim.idle_while_queued,
        "defer_share": sum(r.n_defer > 0 for r in recs) / n_req if n_req else nan,
        "defer_per_req": sum(r.n_defer for r in recs) / n_req if n_req else nan,
        "mean_deferred": float(np.mean([r.deferred for r in started])) if started else nan,
        "arrivals_per_station": per_station,
        # per-request values for comparisons over the requests served under both policies
        "_d_by_rid": {r.rid: r.d for r in started},
        "_w_by_rid": {r.rid: r.w for r in started},
        **{f"n_{o}": c for o, c in counts.items()},
    }
