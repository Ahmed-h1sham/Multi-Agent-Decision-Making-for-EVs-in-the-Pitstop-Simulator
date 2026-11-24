"""Paired runner: every policy runs on the same pre-generated scenario j."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .config import DEFAULT_DRIVERS, DEFAULT_SIM, EVAL_SEED
from .metrics import run_metrics
from .policies import POLICIES
from .scenario import generate
from .sim import run
from .stats import paired_summary

# metric -> which direction is better
METRICS = {
    "mean_w": "lower",
    "p_tau": "lower",
    "mean_d": "lower",
    "completion": "higher",
    "n_abandoned": "lower",
    "est_error": "lower",
    "idle_while_queued": "lower",
    "decision_us": "lower",
    "defer_share": "lower",
    "mean_deferred": "lower",
    "utilisation": "higher",
}
# (first, second): reported as second - first
COMPARISONS = (("A", "B"), ("A", "B_info"), ("B_info", "B"), ("A", "A_d"), ("B", "B_d"))
ALL_POLICIES = ("A", "B_info", "B", "A_d", "B_d")


def run_experiment(levels=("normal", "high", "reduced"), m: int = 30, base_seed: int = EVAL_SEED,
                   policies=ALL_POLICIES, cfg=DEFAULT_SIM) -> list:
    rows = []
    for level in levels:
        for j in range(m):
            sc = generate(level, j, base_seed, sim=cfg)
            digest = sc.digest()
            for key in policies:
                sim = run(sc, POLICIES[key], cfg)
                if sc.digest() != digest:
                    raise RuntimeError("a policy changed the scenario inputs")
                rows.append(run_metrics(sim))
    return rows


def summarize(rows: list) -> dict:
    out = {}
    for level in dict.fromkeys(r["level"] for r in rows):
        lv = [r for r in rows if r["level"] == level]
        by = {}
        for r in lv:
            by.setdefault(r["policy"], {})[r["scenario"]] = r
        out[level] = {}
        for a, b in COMPARISONS:
            if a not in by or b not in by:
                continue
            js = sorted(set(by[a]) & set(by[b]))
            comp = {}
            for metric, better in METRICS.items():
                ya = [by[a][j][metric] for j in js]
                yb = [by[b][j][metric] for j in js]
                s = paired_summary(ya, yb)
                if "diffs" in s:
                    d = np.array(s["diffs"])
                    s["b_better_share"] = float(np.mean(d < 0) if better == "lower" else np.mean(d > 0))
                    s["tied_share"] = float(np.mean(d == 0))
                s["better"] = better
                comp[metric] = s
            comp.update(common_served(by[a], by[b], js))
            out[level][f"{a}_vs_{b}"] = comp
    return out


def common_served(rows_a: dict, rows_b: dict, js: list) -> dict:
    """Delay and wait over the requests that start charging under BOTH policies.

    The plain mean over served requests can favour the policy that serves fewer drivers:
    the drivers who give up are often those who would have waited longest, and they drop
    out of its average. Restricting both policies to the same requests removes that
    selection; the share of requests in the common set is reported with it.
    """
    out = {}
    for key, src in (("d_common", "_d_by_rid"), ("w_common", "_w_by_rid")):
        ya, yb, share = [], [], []
        for j in js:
            da, db = rows_a[j].get(src), rows_b[j].get(src)
            if da is None or db is None:
                ya.append(math.nan), yb.append(math.nan)
                continue
            common = sorted(set(da) & set(db))
            n_req = rows_a[j]["n_req"]
            share.append(len(common) / n_req if n_req else math.nan)
            ya.append(float(np.mean([da[r] for r in common])) if common else math.nan)
            yb.append(float(np.mean([db[r] for r in common])) if common else math.nan)
        s = paired_summary(ya, yb)
        if "diffs" in s:
            d = np.array(s["diffs"])
            s["b_better_share"] = float(np.mean(d < 0))
            s["tied_share"] = float(np.mean(d == 0))
        s["better"] = "lower"
        s["common_share"] = float(np.nanmean(share)) if share else math.nan
        out[key] = s
    return out


RHO_SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
PATIENCE_SWEEP = ((5.0, 30.0), (15.0, 60.0), (30.0, 90.0))
TAU_KEYS = (("p_w5", 5), ("p_w10", 10), ("p_tau", None), ("p_w30", 30))
SENS_METRICS = ("mean_d", "mean_w", "p_tau", "completion", "n_abandoned", "utilisation")


def _means(rows):
    return {k: float(np.nanmean([r[k] for r in rows])) for k in SENS_METRICS}


def _paired(ra, rb, k):
    s = paired_summary([r[k] for r in ra], [r[k] for r in rb])
    s.pop("diffs", None)
    return s


def _run_pair(scenarios, cfg):
    ra = [run_metrics(run(sc, POLICIES["A"], cfg)) for sc in scenarios]
    rb = [run_metrics(run(sc, POLICIES["B"], cfg)) for sc in scenarios]
    js = list(range(len(scenarios)))
    common = common_served(dict(zip(js, ra)), dict(zip(js, rb)), js)
    return {"A": _means(ra), "B": _means(rb), **common,
            **{k: _paired(ra, rb, k) for k in ("mean_d", "mean_w", "p_tau", "completion", "n_abandoned")}}


def sensitivity(rows: list, m: int = 30, base_seed: int = EVAL_SEED, cfg=DEFAULT_SIM,
                drivers=DEFAULT_DRIVERS) -> dict:
    """How the A-vs-B comparison changes with the method values.

    rho        demand from 0.5 to 0.95 (no outages)
    patience   high demand with shorter and longer patience ranges
    tau        the excessive-waiting share at 5, 10, 15 (fixed tau) and 30 min, from rows
    """
    out = {"rho": [], "patience": [], "tau": {}}
    for rho in RHO_SWEEP:
        scs = [generate("custom", j, base_seed, rho=rho, sim=cfg, drivers=drivers) for j in range(m)]
        out["rho"].append({"rho": rho, **_run_pair(scs, cfg)})
    for pr in PATIENCE_SWEEP:
        dm = replace(drivers, patience_range=pr)
        scs = [generate("high", j, base_seed, sim=cfg, drivers=dm) for j in range(m)]
        out["patience"].append({"patience": list(pr), "fixed": tuple(pr) == tuple(drivers.patience_range),
                                **_run_pair(scs, cfg)})
    for level in dict.fromkeys(r["level"] for r in rows):
        by = {}
        for r in rows:
            if r["level"] == level and r["policy"] in ("A", "B"):
                by.setdefault(r["policy"], {})[r["scenario"]] = r
        js = sorted(set(by.get("A", {})) & set(by.get("B", {})))
        out["tau"][level] = [{"tau": t if t is not None else cfg.tau, "fixed": t is None,
                              "A": float(np.nanmean([by["A"][j][k] for j in js])),
                              "B": float(np.nanmean([by["B"][j][k] for j in js])),
                              **{x: v for x, v in paired_summary([by["A"][j][k] for j in js],
                                                                 [by["B"][j][k] for j in js]).items()
                                 if x != "diffs"}} for k, t in TAU_KEYS]
    return out


def pick_typical(rows: list, level: str = "high", metric: str = "mean_w") -> int:
    """Scenario whose A-vs-B difference is closest to the median difference.

    Used to choose the scenario shown in the replay, so the replay is a typical case,
    not the most flattering one.
    """
    by = {}
    for r in rows:
        if r["level"] == level:
            by.setdefault(r["scenario"], {})[r["policy"]] = r[metric]
    diffs = {j: v["B"] - v["A"] for j, v in by.items()
             if "A" in v and "B" in v and math.isfinite(v["A"]) and math.isfinite(v["B"])}
    med = float(np.median(list(diffs.values())))
    return min(diffs, key=lambda j: (abs(diffs[j] - med), j))


def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


def export_replay(level: str, index: int, base_seed: int = EVAL_SEED, policies=("A", "B"), cfg=DEFAULT_SIM) -> dict:
    """Everything the replay page needs to redraw one scenario, from real runs."""
    sc = generate(level, index, base_seed, sim=cfg)
    out = {
        "scenario": {"level": level, "index": index, "seed": base_seed, "digest": sc.digest(),
                     "rho": sc.rho, "rate_per_hour": sc.arrival_rate_per_hour,
                     "t_warm": cfg.t_warm, "t_last": cfg.t_arrivals_end, "t_end": cfg.t_end, "tau": cfg.tau,
                     "tolerance": cfg.booking_tolerance, "area_km": cfg.area_km},
        "stations": [{"sid": s.sid, "name": s.name, "x": s.x, "y": s.y, "type": s.charger_type,
                      "c": s.n_chargers} for s in sc.stations],
        "outages": [{"sid": o.sid, "n": o.n_down, "t0": o.t_start, "t1": o.t_end} for o in sc.outages],
        "runs": {},
    }
    req = {r.rid: r for r in sc.requests}
    for key in policies:
        sim = run(sc, POLICIES[key], cfg)
        recs = []
        for rec in sim.recs.values():
            r = req[rec.rid]
            recs.append({
                "rid": rec.rid, "tn": _r(rec.t_need), "x": r.x, "y": r.y, "veh": r.vehicle,
                "soc0": _r(r.soc0, 3), "sid": rec.sid, "ta": _r(rec.t_arr), "tae": _r(rec.t_arr_est),
                "ts": _r(rec.t_start), "te": _r(rec.t_end), "tl": _r(rec.t_leave),
                "est": _r(rec.est_start), "dur": _r(rec.dur), "kwh": _r(rec.energy_kwh, 1),
                "out": rec.outcome, "bseq": rec.bseq, "ent": rec.entitled, "nd": rec.n_defer,
                "pat": _r(r.patience),
                "seen": {str(k): _r(v) for k, v in (rec.seen or {}).items()},
            })
        m = run_metrics(sim)
        out["runs"][key] = {
            "label": POLICIES[key].label,
            "records": recs,
            "metrics": {k: (_r(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
            "board_log": [[_r(t), e, rid, sid, bseq] for t, e, rid, sid, bseq in sim.board.log]
            if sim.board is not None else [],
        }
    return out
