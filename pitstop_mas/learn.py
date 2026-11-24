"""Parameter-shared PPO for the PettingZoo environment, in NumPy.

Every EV uses the same policy network (parameter sharing) and decides from its own
observation (decentralised execution); the value function also sees only the EV's own
observation, so this is independent PPO with shared parameters. It is the learned
counterpart of the rule-based policies: same environment, same information, same actions
(a station or defer), and the same objective, the EV's own access delay d.

Why not Stable-Baselines3: SB3 trains one agent whose reward follows its own action.
Here many EVs act in turn and each EV's reward arrives only when its access is settled,
after other EVs have acted, so transitions are collected per EV and advantages are
computed along each EV's own decisions (generalised advantage estimation, gamma = 1).

The PPO update follows the usual clipped-surrogate recipe (Schulman et al., 2017):
clipped policy ratio, value loss, entropy bonus, Adam, global gradient-norm clipping,
minibatch epochs. Illegal actions (masked) get zero probability.

Training and evaluation use different scenarios: training draws scenarios from base
seed TRAIN_SEED, evaluation uses the experiment's scenarios (base seed EVAL_SEED).
"""
from __future__ import annotations

import csv
import json
import math
import time

import numpy as np

from .config import DEFAULT_SIM, EVAL_SEED, TRAIN_SEED
from .env import PitstopEnv
from .metrics import run_metrics
from .scenario import generate

LEVELS = ("normal", "high", "reduced")


# ---- network ------------------------------------------------------------------------
def _orthogonal(rng, shape, gain):
    a = rng.standard_normal(shape if shape[0] >= shape[1] else shape[::-1])
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if shape[0] < shape[1]:
        q = q.T
    return gain * q[: shape[0], : shape[1]]


class MLP:
    """tanh MLP with a linear output layer; row-vector convention h = x @ W + b."""

    def __init__(self, sizes, rng, out_gain):
        self.p = {}
        self.n = len(sizes) - 1
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]), 1):
            gain = out_gain if i == self.n else math.sqrt(2.0)
            self.p[f"W{i}"] = _orthogonal(rng, (a, b), gain)
            self.p[f"b{i}"] = np.zeros(b)

    def forward(self, x):
        cache, h = [x], x
        for i in range(1, self.n + 1):
            z = h @ self.p[f"W{i}"] + self.p[f"b{i}"]
            h = np.tanh(z) if i < self.n else z
            cache.append(h)
        return h, cache

    def backward(self, cache, dout):
        g, d = {}, dout
        for i in range(self.n, 0, -1):
            g[f"W{i}"] = cache[i - 1].T @ d
            g[f"b{i}"] = d.sum(0)
            if i > 1:
                d = (d @ self.p[f"W{i}"].T) * (1.0 - cache[i - 1] ** 2)
        return g


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-5):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def _masked_log_softmax(logits, mask):
    z = np.where(mask > 0, logits, -1e9)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


class SharedPolicy:
    """One actor and one critic shared by all EVs."""

    def __init__(self, n_obs, n_act, seed=0, hidden=64):
        rng = np.random.default_rng(seed)
        self.n_obs, self.n_act, self.hidden = n_obs, n_act, hidden
        self.actor = MLP([n_obs, hidden, hidden, n_act], rng, 0.01)
        self.critic = MLP([n_obs, hidden, hidden, 1], rng, 1.0)
        self.meta = {}

    def probs(self, obs, mask):
        logits, _ = self.actor.forward(np.atleast_2d(obs))
        return np.exp(_masked_log_softmax(logits, np.atleast_2d(mask)))

    def act(self, obs, mask, rng=None, greedy=False):
        p = self.probs(obs, mask)[0]
        a = int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p / p.sum()))
        v = float(self.critic.forward(np.atleast_2d(obs))[0][0, 0])
        return a, float(np.log(p[a])), v

    def params(self):
        return {**{f"a_{k}": v for k, v in self.actor.p.items()},
                **{f"c_{k}": v for k, v in self.critic.p.items()}}

    def save(self, path):
        np.savez(path, meta=json.dumps({"n_obs": self.n_obs, "n_act": self.n_act,
                                        "hidden": self.hidden, **self.meta}), **self.params())

    @classmethod
    def load(cls, path):
        f = np.load(path, allow_pickle=False)
        meta = json.loads(str(f["meta"]))
        pol = cls(meta["n_obs"], meta["n_act"], hidden=meta["hidden"])
        for k in pol.actor.p:
            pol.actor.p[k] = f[f"a_{k}"]
        for k in pol.critic.p:
            pol.critic.p[k] = f[f"c_{k}"]
        pol.meta = {k: v for k, v in meta.items() if k not in ("n_obs", "n_act", "hidden")}
        return pol


# ---- PPO loss and gradients -----------------------------------------------------------
def ppo_loss_and_grads(pol: SharedPolicy, b: dict, clip=0.2, ent_coef=0.01, vf_coef=0.5):
    """Clipped PPO loss for a minibatch and its exact gradients.

    b: obs (B, n_obs), mask (B, n_act), act (B,), logp (B,) old log-probs,
       adv (B,) advantages (already normalised), ret (B,) value targets.
    """
    obs, mask, act = b["obs"], b["mask"], b["act"]
    B = len(act)
    rows = np.arange(B)
    logits, ca = pol.actor.forward(obs)
    logp_all = _masked_log_softmax(logits, mask)
    p = np.exp(logp_all)
    logp = logp_all[rows, act]
    ratio = np.exp(logp - b["logp"])
    adv = b["adv"]
    s1, s2 = ratio * adv, np.clip(ratio, 1 - clip, 1 + clip) * adv
    pg_loss = -np.mean(np.minimum(s1, s2))
    plogp = np.where(mask > 0, p * logp_all, 0.0)
    entropy = -plogp.sum(1)

    use = (s1 <= s2).astype(float)                       # unclipped term is the active one
    g_logp = -(adv * ratio * use) / B
    onehot = np.zeros_like(p)
    onehot[rows, act] = 1.0
    dz = g_logp[:, None] * (onehot - p)
    dz += (ent_coef / B) * np.where(mask > 0, p * (logp_all + entropy[:, None]), 0.0)
    dz = np.where(mask > 0, dz, 0.0)
    ga = pol.actor.backward(ca, dz)

    v, cc = pol.critic.forward(obs)
    v = v[:, 0]
    v_loss = 0.5 * np.mean((v - b["ret"]) ** 2)
    gc = pol.critic.backward(cc, (vf_coef * (v - b["ret"]) / B)[:, None])

    loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss
    grads = {**{f"a_{k}": g for k, g in ga.items()}, **{f"c_{k}": g for k, g in gc.items()}}
    stats = {"loss": float(loss), "pg_loss": float(pg_loss), "v_loss": float(v_loss),
             "entropy": float(entropy.mean()),
             "approx_kl": float(np.mean((ratio - 1) - np.log(ratio))),
             "clipfrac": float(np.mean(np.abs(ratio - 1) > clip))}
    return loss, grads, stats


def gae(rewards, values, gamma=1.0, lam=0.95):
    """Advantages along one EV's decisions; the reward arrives at its last decision."""
    adv = np.zeros(len(rewards))
    last = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        next_v = values[t + 1] if t + 1 < len(rewards) else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv


# ---- rollouts ----------------------------------------------------------------------------
def play_episode(pol, e: PitstopEnv, index, rng=None, greedy=False, record=None):
    """Run one episode. If record is a list, per-EV trajectories are appended to it."""
    e.reset(options={"index": index})
    traj = {}
    for agent in e.agent_iter():
        obs, rew, term, trunc, _info = e.last()
        if term or trunc:
            t = traj.pop(agent, None)
            if t is not None and record is not None:
                t["rew"][-1] = rew
                record.append(t)
            e.step(None)
            continue
        o, m = obs["observation"].astype(np.float64), obs["action_mask"]
        a, lp, v = pol.act(o, m, rng, greedy=greedy)
        t = traj.setdefault(agent, {"obs": [], "mask": [], "act": [], "logp": [], "val": [], "rew": []})
        t["obs"].append(o), t["mask"].append(m), t["act"].append(a)
        t["logp"].append(lp), t["val"].append(v), t["rew"].append(0.0)
        e.step(a)
    return run_metrics(e.sim)


def training_scenarios(base_seed=TRAIN_SEED, cfg=DEFAULT_SIM):
    """Scenario i: demand level cycles normal / high / reduced; never an evaluation scenario."""
    return lambda i: generate(LEVELS[i % len(LEVELS)], i, base_seed, sim=cfg)


def train(setting="B_d", updates=250, episodes_per_update=8, seed=0, lr=3e-4, epochs=4,
          minibatch=256, clip=0.2, ent_coef=0.01, vf_coef=0.5, gamma=1.0, lam=0.95,
          max_grad_norm=0.5, hidden=64, cfg=DEFAULT_SIM, log=None, curve_path=None):
    rng = np.random.default_rng(seed)
    e = PitstopEnv(setting=setting, cfg=cfg, scenario_fn=training_scenarios(cfg=cfg))
    n_obs = e.observation_space(e.possible_agents[0])["observation"].shape[0]
    pol = SharedPolicy(n_obs, e.K + 1, seed=seed, hidden=hidden)
    pol.meta = {"setting": setting, "seed": seed, "updates": updates,
                "episodes_per_update": episodes_per_update, "lr": lr, "ent_coef": ent_coef,
                "clip": clip, "gamma": gamma, "lam": lam, "train_seed": TRAIN_SEED,
                "station_ids": e.station_ids}
    opt = Adam(pol.params(), lr)
    curve, episode, t0 = [], 0, time.time()
    for u in range(1, updates + 1):
        trajs, mets = [], []
        for _ in range(episodes_per_update):
            mets.append(play_episode(pol, e, episode, rng, record=trajs))
            episode += 1
        obs = np.concatenate([np.array(t["obs"]) for t in trajs])
        mask = np.concatenate([np.array(t["mask"]) for t in trajs])
        act = np.concatenate([np.array(t["act"]) for t in trajs])
        logp = np.concatenate([np.array(t["logp"]) for t in trajs])
        val = np.concatenate([np.array(t["val"]) for t in trajs])
        adv = np.concatenate([gae(np.array(t["rew"]), np.array(t["val"]), gamma, lam) for t in trajs])
        ret = adv + val
        n = len(act)
        stats = []
        for _ in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, minibatch):
                idx = order[s:s + minibatch]
                a_ = adv[idx]
                batch = {"obs": obs[idx], "mask": mask[idx], "act": act[idx], "logp": logp[idx],
                         "adv": (a_ - a_.mean()) / (a_.std() + 1e-8), "ret": ret[idx]}
                _loss, grads, st = ppo_loss_and_grads(pol, batch, clip, ent_coef, vf_coef)
                norm = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
                if norm > max_grad_norm:
                    grads = {k: g * (max_grad_norm / norm) for k, g in grads.items()}
                params = pol.params()
                opt.step(params, grads)
                for k in pol.actor.p:
                    pol.actor.p[k] = params[f"a_{k}"]
                for k in pol.critic.p:
                    pol.critic.p[k] = params[f"c_{k}"]
                stats.append(st)
        ep_rew = float(np.mean([t["rew"][-1] for t in trajs]))
        row = {"update": u, "episodes": episode, "mean_reward": ep_rew,
               "mean_d": float(np.nanmean([m["mean_d"] for m in mets])),
               "completion": float(np.nanmean([m["completion"] for m in mets])),
               "defer_share": float(np.nanmean([m["defer_share"] for m in mets])),
               "entropy": float(np.mean([s["entropy"] for s in stats])),
               "approx_kl": float(np.mean([s["approx_kl"] for s in stats])),
               "seconds": round(time.time() - t0, 1)}
        curve.append(row)
        if log and (u % 10 == 0 or u == 1 or u == updates):
            log(f"  update {u:4d}  reward {ep_rew:7.3f}  d {row['mean_d']:6.2f} min  "
                f"C {row['completion'] * 100:5.1f} %  defer {row['defer_share'] * 100:4.1f} %  "
                f"entropy {row['entropy']:.3f}  {row['seconds']:6.1f} s")
    if curve_path:
        with open(curve_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(curve[0]))
            w.writeheader()
            w.writerows(curve)
    pol.meta["train_seconds"] = round(time.time() - t0, 1)
    return pol, curve


REPORT_METRICS = ("mean_d", "mean_w", "p_tau", "completion", "n_abandoned", "defer_share", "est_error")


def _mean_rows(rows_by_seed):
    """Average the numeric metrics of several seeds, scenario by scenario."""
    out = []
    for rows in zip(*rows_by_seed):
        base = dict(rows[0])
        for k in REPORT_METRICS + ("idle_while_queued", "mean_deferred", "n_complete", "n_started"):
            vals = [r[k] for r in rows]
            base[k] = float(np.mean(vals)) if all(np.isfinite(vals)) else math.nan
        base["policy"] = "PPO"
        out.append(base)
    return out


def rule_reference(setting, base_seed=TRAIN_SEED, n=60, cfg=DEFAULT_SIM):
    """Mean d of the matching rule policy on training scenarios (reference line for curves)."""
    from .policies import POLICIES
    from .sim import run
    fn = training_scenarios(base_seed, cfg)
    return float(np.nanmean([run_metrics(run(fn(100000 + i), POLICIES[setting], cfg))["mean_d"] for i in range(n)]))


def ppo_report(model_paths, m=30, base_seed=EVAL_SEED, levels=LEVELS, cfg=DEFAULT_SIM, rule_rows=None):
    """Evaluate saved models on the evaluation scenarios and compare them with the rules."""
    from pathlib import Path

    from .experiment import run_experiment
    from .stats import paired_summary

    paths = sorted(Path(p) for p in model_paths)
    pols = [SharedPolicy.load(p) for p in paths]
    setting = pols[0].meta.get("setting", "B_d")
    rules = tuple(dict.fromkeys(("A", "B", setting)))
    if rule_rows is None:
        rule_rows = run_experiment(levels=levels, m=m, base_seed=base_seed, policies=rules, cfg=cfg)
    by_seed = [evaluate(p, setting, levels, m, base_seed, cfg, label=f"PPO_s{p.meta.get('seed')}") for p in pols]
    mean_rows = _mean_rows(by_seed)
    report = {"setting": setting, "rules": list(rules), "m": m,
              "seeds": [p.meta.get("seed") for p in pols],
              "train": [{k: p.meta.get(k) for k in ("seed", "updates", "episodes_per_update", "lr",
                                                     "ent_coef", "train_seconds", "train_seed")} for p in pols],
              "levels": {}, "curves": []}
    for level in levels:
        def pick(rows, key=None):
            return [r for r in rows if r["level"] == level and (key is None or r["policy"] == key)]
        ppo = pick(mean_rows)
        lv = {"means": {}, "vs": {}, "seed_d_vs_rule": []}
        for key in rules:
            rr = pick(rule_rows, key)
            lv["means"][key] = {k: float(np.nanmean([r[k] for r in rr])) for k in REPORT_METRICS}
            comp = {}
            for k in REPORT_METRICS:
                s = paired_summary([r[k] for r in rr], [r[k] for r in ppo])
                s.pop("diffs", None)
                comp[k] = s
            lv["vs"][key] = comp
        lv["means"]["PPO"] = {k: float(np.nanmean([r[k] for r in ppo])) for k in REPORT_METRICS}
        rr = pick(rule_rows, setting)
        for p, rows in zip(pols, by_seed):
            s = paired_summary([r["mean_d"] for r in rr], [r["mean_d"] for r in pick(rows)])
            lv["seed_d_vs_rule"].append({"seed": p.meta.get("seed"), "mean": s.get("mean_b"),
                                         "diff": s.get("mean_diff"), "ci_low": s.get("ci_low"),
                                         "ci_high": s.get("ci_high"), "p": s.get("p_paired")})
        report["levels"][level] = lv
    for p, path in zip(pols, paths):
        curve = path.with_name(path.stem + "_curve.csv")
        if curve.exists():
            with open(curve) as fh:
                rows = list(csv.DictReader(fh))
            report["curves"].append({"seed": p.meta.get("seed"),
                                     "update": [int(r["update"]) for r in rows],
                                     "mean_d": [float(r["mean_d"]) for r in rows],
                                     "defer_share": [float(r["defer_share"]) for r in rows]})
    ref = paths[0].with_name(f"rule_reference_{setting}.json")
    if ref.exists():
        report["rule_train_d"] = json.loads(ref.read_text())["mean_d"]
    report["rows"] = by_seed
    return report


def evaluate(pol, setting=None, levels=LEVELS, m=30, base_seed=EVAL_SEED, cfg=DEFAULT_SIM, label=None):
    """Greedy policy on the evaluation scenarios; one run_metrics row per scenario."""
    setting = setting or pol.meta.get("setting", "B_d")
    rows = []
    for level in levels:
        e = PitstopEnv(setting=setting, level=level, base_seed=base_seed, cfg=cfg)
        for j in range(m):
            row = play_episode(pol, e, j, greedy=True)
            row["policy"] = label or f"PPO_{setting}"
            rows.append(row)
    return rows
