"""Train the parameter-shared PPO policy and compare it with the rule-based policies.

    python train_ppo.py                         # setting B_d, seeds 0 1 2, 800 updates each
    python train_ppo.py --setting A_d --seeds 0  # without the reservation board
    python train_ppo.py --updates 50 --seeds 0   # quick check

Training uses scenarios from base seed 1000 (never the evaluation scenarios). Each trained
policy is then run greedily on the 30 x 3 evaluation scenarios (base seed 2027) and
compared, scenario by scenario, with the rules A, B and the rule of the same setting.

Outputs in results/ppo/: <setting>_s<seed>.npz (model), <setting>_s<seed>_curve.csv
(training curve), rule_reference_<setting>.json, ppo_report_<setting>.json.
run_demo.py picks the saved models up automatically.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pitstop_mas.config import EVAL_SEED
from pitstop_mas.learn import ppo_report, rule_reference, train

HERE = Path(__file__).resolve().parent


def _f(x, nd=2):
    return "  n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", default="B_d", choices=["A", "B_info", "B", "A_d", "B_d"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--updates", type=int, default=800)
    ap.add_argument("--episodes", type=int, default=8, help="episodes per update")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ent", type=float, default=0.003, help="entropy coefficient")
    ap.add_argument("--m", type=int, default=30, help="evaluation scenarios per level")
    ap.add_argument("--out", default="results/ppo")
    args = ap.parse_args(argv)
    out = HERE / args.out
    out.mkdir(parents=True, exist_ok=True)

    ref = rule_reference(args.setting)
    (out / f"rule_reference_{args.setting}.json").write_text(json.dumps({"setting": args.setting, "mean_d": ref}))
    print(f"Rule {args.setting} on training scenarios: mean d {ref:.2f} min")

    paths = []
    for seed in args.seeds:
        print(f"\n== Training PPO, setting {args.setting}, seed {seed} ==")
        stem = out / f"{args.setting}_s{seed}"
        pol, _ = train(setting=args.setting, updates=args.updates, episodes_per_update=args.episodes,
                       seed=seed, lr=args.lr, ent_coef=args.ent, log=print,
                       curve_path=stem.with_name(stem.name + "_curve.csv"))
        pol.save(stem.with_suffix(".npz"))
        paths.append(stem.with_suffix(".npz"))

    print(f"\n== Evaluation on {args.m} scenarios per level (base seed {EVAL_SEED}) ==")
    rep = ppo_report(paths, m=args.m)
    (out / f"ppo_report_{args.setting}.json").write_text(json.dumps(
        {k: v for k, v in rep.items() if k != "rows"}, indent=1, default=str))
    for level, lv in rep["levels"].items():
        print(f"\n  [{level}]  mean over {len(rep['seeds'])} seed(s)")
        print("  policy   d (min)  w (min)  C (%)   left   defer (%)")
        for key, mm in lv["means"].items():
            print(f"  {key:7s} {_f(mm['mean_d']):>8} {_f(mm['mean_w']):>8} {_f(mm['completion'] * 100, 1):>6} "
                  f"{_f(mm['n_abandoned']):>6} {_f(mm['defer_share'] * 100, 1):>8}")
        for key, comp in lv["vs"].items():
            s = comp["mean_d"]
            print(f"  PPO - {key:5s} d: {_f(s['mean_diff'])} [{_f(s['ci_low'])}, {_f(s['ci_high'])}]  p {s['p_paired']:.4f}")


if __name__ == "__main__":
    main()
