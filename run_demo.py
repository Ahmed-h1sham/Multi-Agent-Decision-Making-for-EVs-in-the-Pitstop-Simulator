"""Run the whole demo: validation, paired experiments, defer action, replay export,
PettingZoo check, demo page.

    python run_demo.py              # tests + 30 scenarios per demand level (about 20 s)
    python run_demo.py --m 10       # quicker
    python run_demo.py --skip-tests # skip the pytest step

Outputs go to results/ (CSV + JSON) and results/pitstop_demo.html (open in a browser).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from pitstop_mas.config import DEFAULT_DRIVERS, DEFAULT_SIM, DEMAND_LEVELS, EVAL_SEED, STATIONS
from pitstop_mas.experiment import (ALL_POLICIES, export_replay, pick_typical, run_experiment,
                                    sensitivity, summarize)
from pitstop_mas.metrics import run_metrics
from pitstop_mas.policies import INDEPENDENT, POLICIES, RESERVATION
from pitstop_mas.scenario import generate, toy_herding
from pitstop_mas.sim import run
from pitstop_mas.validation import erlang_table

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "pitstop_mas" / "replay_template.html"


def _fmt(x, nd=2):
    return "  n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _clean(o):
    """JSON-safe copy: NaN and infinities become null (undefined, as the thesis reports them)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):          # numpy scalars
        return _clean(o.item())
    return o


def toy_check() -> dict:
    """Three EVs, two stations: the background-chapter herding figure, as a run."""
    sc = toy_herding()
    a, b = run(sc, INDEPENDENT), run(sc, RESERVATION)
    names = {0: "A", 1: "B"}
    return {
        "independent": [names[a.recs[i].sid] for i in range(3)],
        "board": [names[b.recs[i].sid] for i in range(3)],
        "w_independent": run_metrics(a)["mean_w"],
        "w_board": run_metrics(b)["mean_w"],
    }


def env_check(level: str, index: int, seed: int) -> dict:
    """Drive one scenario through the PettingZoo environment.

    1. Every rule-based policy, deciding through the environment, must give exactly the
       same run as the direct simulation (PettingZoo adds no behaviour).
    2. Random agents restricted to legal actions show that any controller can drive it.
    """
    try:
        from pitstop_mas.env import PitstopEnv, rule_action
    except ImportError:
        return {"available": False}
    sc = generate(level, index, seed)
    out = {"available": True, "level": level, "index": index, "settings": []}
    for key, pol in POLICIES.items():
        e = PitstopEnv(setting=key, level=level, base_seed=seed)
        e.reset(options={"index": index})
        decisions, agents = 0, set()
        for agent in e.agent_iter():
            agents.add(agent)
            _o, _r, term, trunc, _i = e.last()
            if term or trunc:
                e.step(None)
            else:
                decisions += 1
                e.step(rule_action(e))
        via_env, direct = run_metrics(e.sim), run_metrics(run(sc, pol))
        via_env.pop("decision_us"), direct.pop("decision_us")
        out["settings"].append({"setting": key, "identical": via_env == direct,
                                "decisions": decisions, "agents": len(agents)})
    rng = np.random.default_rng(seed)
    e = PitstopEnv(setting="B_d", level=level, base_seed=seed)
    e.reset(options={"index": index})
    for _agent in e.agent_iter():
        _o, _r, term, trunc, _i = e.last()
        if term or trunc:
            e.step(None)
        else:
            mask = e.observe(e.agent_selection)["action_mask"]
            e.step(int(rng.choice(np.flatnonzero(mask))))
    m, rule = run_metrics(e.sim), run_metrics(run(sc, POLICIES["B_d"]))
    out["random"] = {"setting": "B_d", "mean_w": m["mean_w"], "mean_d": m["mean_d"],
                     "completion": m["completion"], "defer_share": m["defer_share"],
                     "rule_mean_d": rule["mean_d"], "rule_completion": rule["completion"]}
    return out


def ppo_check(m: int, seed: int, rule_rows: list) -> dict:
    """Evaluate every saved PPO model on the evaluation scenarios, grouped by setting."""
    models = sorted(p for p in (HERE / "results" / "ppo").glob("*_s*.npz"))
    if not models or seed != EVAL_SEED:
        return {}
    from pitstop_mas.learn import ppo_report
    groups = {}
    for p in models:
        groups.setdefault(p.stem.rsplit("_s", 1)[0], []).append(p)
    out = {}
    for setting, paths in groups.items():
        rep = ppo_report(paths, m=m, base_seed=seed, rule_rows=rule_rows)
        rep.pop("rows")
        out[setting] = rep
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=30, help="scenarios per demand level")
    ap.add_argument("--seed", type=int, default=EVAL_SEED, help="base seed of the evaluation scenarios")
    ap.add_argument("--out", default="results")
    ap.add_argument("--erlang-n", type=int, default=150_000, help="customers per M/M/c validation run")
    ap.add_argument("--replay-level", default="high", choices=list(DEMAND_LEVELS))
    ap.add_argument("--skip-tests", action="store_true", help="do not run the validation tests")
    args = ap.parse_args(argv)
    out = HERE / args.out
    out.mkdir(exist_ok=True)

    t0 = time.time()
    tests = []
    if not args.skip_tests:
        print("== 0. Validation tests (pytest) ==")
        xml = out / "tests.xml"
        rc = subprocess.call([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                              f"--junitxml={xml}", str(HERE / "tests")], cwd=HERE)
        for case in ET.parse(xml).getroot().iter("testcase"):
            failed = case.find("failure") is not None or case.find("error") is not None
            tests.append({"name": case.get("name"), "passed": not failed,
                          "seconds": float(case.get("time", 0))})
        if rc != 0:
            print("Some validation tests failed; see above. Continuing so the page shows them.")

    print("\n== 1. Queue core vs Erlang C (single M/M/c station) ==")
    erl = erlang_table(n=args.erlang_n)
    print("  c   rho   Erlang C Wq   simulated Wq   95% CI")
    for v in erl:
        print(f"  {v['c']}   {v['rho']:.1f}   {v['analytic_wq']:10.3f}   {v['sim_wq']:12.3f}   "
              f"[{v['ci_low']:.3f}, {v['ci_high']:.3f}]")

    print("\n== 2. Herding toy case (3 EVs, stations A and B) ==")
    toy = toy_check()
    print(f"  independent choice : {', '.join(toy['independent'])}   mean wait {toy['w_independent']:.1f} min")
    print(f"  reservation board  : {', '.join(toy['board'])}   mean wait {toy['w_board']:.1f} min")

    print(f"\n== 3. Paired experiment: {args.m} scenarios x 3 demand levels x {len(ALL_POLICIES)} policies ==")
    rows = run_experiment(m=args.m, base_seed=args.seed)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    df["arrivals_per_station"] = df["arrivals_per_station"].map(lambda v: "|".join(map(str, v)))
    df.to_csv(out / "runs.csv", index=False)
    summary = summarize(rows)
    labels = {"mean_w": "mean wait w (min)", "p_tau": f"share w > {DEFAULT_SIM.tau:.0f} min",
              "mean_d": "access delay d (min)", "completion": "completion C",
              "n_abandoned": "abandoned (count)"}
    for level, comps in summary.items():
        c = comps["A_vs_B"]
        print(f"\n  [{level}]  A = independent, B = reservation board")
        print("  metric                     A        B     B-A    95% CI              p (paired)  B better")
        for k, lab in labels.items():
            s = c[k]
            print(f"  {lab:24s} {_fmt(s['mean_a']):>7} {_fmt(s['mean_b']):>7} {_fmt(s['mean_diff']):>7}  "
                  f"[{_fmt(s['ci_low'])}, {_fmt(s['ci_high'])}]  {s['p_paired']:10.4f}   "
                  f"{round(s['b_better_share'] * s['m'])}/{s['m']}")
        for k, lab in (("d_common", "d, served under both"), ("w_common", "w, served under both")):
            s = c[k]
            print(f"  {lab:24s} {_fmt(s['mean_a']):>7} {_fmt(s['mean_b']):>7} {_fmt(s['mean_diff']):>7}  "
                  f"[{_fmt(s['ci_low'])}, {_fmt(s['ci_high'])}]  {s['p_paired']:10.4f}   "
                  f"{round(s['b_better_share'] * s['m'])}/{s['m']}   ({_fmt(s['common_share'] * 100, 0)} % of requests)")
        i = comps["A_vs_B_info"]["mean_w"]
        print(f"  board info only (B_info) vs A, mean wait: {_fmt(i['mean_diff'])} min "
              f"[{_fmt(i['ci_low'])}, {_fmt(i['ci_high'])}]")

    print("\n== 4. Defer action: wait where you are and decide again (A -> A_d, B -> B_d) ==")
    dlabels = {"mean_w": "mean wait w (min)", "mean_d": "access delay d (min)",
               "p_tau": f"share w > {DEFAULT_SIM.tau:.0f} min", "completion": "completion C",
               "n_abandoned": "abandoned (count)"}
    for level, comps in summary.items():
        for base, dd in (("A", "A_d"), ("B", "B_d")):
            c = comps[f"{base}_vs_{dd}"]
            print(f"\n  [{level}] {base} -> {dd}: {_fmt(c['defer_share']['mean_b'] * 100, 0)} % of EVs defer, "
                  f"{_fmt(c['mean_deferred']['mean_b'], 2)} min deferred per started EV")
            for k, lab in dlabels.items():
                s = c[k]
                print(f"    {lab:24s} {_fmt(s['mean_a']):>7} {_fmt(s['mean_b']):>7} {_fmt(s['mean_diff']):>7}  "
                      f"[{_fmt(s['ci_low'])}, {_fmt(s['ci_high'])}]  p {s['p_paired']:.4f}")

    pick = pick_typical(rows, args.replay_level, "completion")
    replay = export_replay(args.replay_level, pick, args.seed)
    print(f"\n== 5. Replay: {args.replay_level} demand, scenario {pick} (median A-vs-B difference) ==")

    print("\n== 6. PettingZoo interface on the same simulator ==")
    envc = env_check(args.replay_level, pick, args.seed)
    if not envc["available"]:
        print("  pettingzoo is not installed; skipped (pip install pettingzoo gymnasium)")
    else:
        print("  setting  decisions  identical to direct run")
        for s in envc["settings"]:
            print(f"  {s['setting']:7s}  {s['decisions']:9d}  {'yes' if s['identical'] else 'NO'}")
        rnd = envc["random"]
        print(f"  random masked agents on B_d: mean d {_fmt(rnd['mean_d'])} min, completion "
              f"{_fmt(rnd['completion'] * 100, 1)} % (rule B_d: {_fmt(rnd['rule_mean_d'])} min, "
              f"{_fmt(rnd['rule_completion'] * 100, 1)} %)")

    print("\n== 7. Sensitivity of the A-vs-B comparison to the method values ==")
    sens = sensitivity(rows, m=args.m, base_seed=args.seed)
    print("  demand rho   util A   d: A -> B   (B-A [95% CI])          C: A -> B")
    for s in sens["rho"]:
        d, c = s["mean_d"], s["completion"]
        print(f"  {s['rho']:.2f}        {_fmt(s['A']['utilisation'], 2)}    {_fmt(d['mean_a'])} -> {_fmt(d['mean_b'])}  "
              f"({_fmt(d['mean_diff'])} [{_fmt(d['ci_low'])}, {_fmt(d['ci_high'])}])   "
              f"{_fmt(c['mean_a'] * 100, 1)} -> {_fmt(c['mean_b'] * 100, 1)} %")
    print("  patience (high demand)   d: A -> B (B-A [CI])                left: A -> B")
    for s in sens["patience"]:
        d, a = s["mean_d"], s["n_abandoned"]
        print(f"  {s['patience'][0]:.0f}-{s['patience'][1]:.0f} min{' (fixed)' if s['fixed'] else '        '}   "
              f"{_fmt(d['mean_a'])} -> {_fmt(d['mean_b'])} ({_fmt(d['mean_diff'])} [{_fmt(d['ci_low'])}, {_fmt(d['ci_high'])}])"
              f"   {_fmt(a['mean_a'])} -> {_fmt(a['mean_b'])}")
    for level, ts in sens["tau"].items():
        print(f"  [{level}] share of waits over tau, A -> B: " + "   ".join(
            f"{t['tau']:.0f} min {_fmt(t['A'] * 100, 1)}->{_fmt(t['B'] * 100, 1)} %" for t in ts))
    (out / "sensitivity.json").write_text(json.dumps(_clean(sens), indent=1))

    print("\n== 8. Learned policy (parameter-shared PPO, models in results/ppo/) ==")
    ppo = ppo_check(args.m, args.seed, rows) if envc["available"] else {}
    if not ppo:
        print("  no trained models found; run  python train_ppo.py  to train them")
    for setting, rep in ppo.items():
        for level, lv in rep["levels"].items():
            s = lv["vs"][setting]["mean_d"]
            print(f"  [{level:8s}] PPO ({setting}, {len(rep['seeds'])} seeds) d {_fmt(lv['means']['PPO']['mean_d'])} min"
                  f" vs rule {setting} {_fmt(lv['means'][setting]['mean_d'])}: {_fmt(s['mean_diff'])} "
                  f"[{_fmt(s['ci_low'])}, {_fmt(s['ci_high'])}]  p {s['p_paired']:.4f}")

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "config": {"m": args.m, "seed": args.seed, "tau": DEFAULT_SIM.tau,
                   "t_warm": DEFAULT_SIM.t_warm, "t_obs": DEFAULT_SIM.t_obs, "t_end": DEFAULT_SIM.t_end,
                   "patience": list(DEFAULT_DRIVERS.patience_range),
                   "tolerance": DEFAULT_SIM.booking_tolerance,
                   "defer_step": DEFAULT_SIM.defer_step, "max_defer": DEFAULT_SIM.max_defer,
                   "levels": {k: {"rho": v.rho, "outage_chargers": v.outage_chargers,
                                  "outage_minutes": v.outage_minutes} for k, v in DEMAND_LEVELS.items()},
                   "stations": [s.__dict__ for s in STATIONS]},
        "erlang": erl,
        "toy": toy,
        "tests": tests,
        "env_check": envc,
        "ppo": ppo,
        "sensitivity": sens,
        "summary": {lv: {cmp: {k: {kk: vv for kk, vv in s.items()} for k, s in comps2.items()}
                         for cmp, comps2 in comps.items()} for lv, comps in summary.items()},
        "replay": replay,
    }
    for name, obj in (("summary.json", summary), ("validation.json", {"erlang": erl, "toy": toy}),
                      ("replay.json", replay), ("env_check.json", envc)):
        (out / name).write_text(json.dumps(obj, indent=1, default=str))
    if TEMPLATE.exists():
        html = TEMPLATE.read_text(encoding="utf-8")
        data = json.dumps(_clean(payload), separators=(",", ":"), allow_nan=False)
        (out / "pitstop_demo.html").write_text(html.replace("/*__DATA__*/null", data), encoding="utf-8")
        print(f"  demo page: {out / 'pitstop_demo.html'}")
    print(f"\nDone in {time.time() - t0:.1f} s. Results in {out}/")


if __name__ == "__main__":
    main()
