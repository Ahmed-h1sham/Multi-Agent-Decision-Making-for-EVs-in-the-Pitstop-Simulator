"""Tests for the parameter-shared PPO (pitstop_mas/learn.py)."""
import numpy as np
import pytest

pytest.importorskip("pettingzoo")

from pitstop_mas.learn import SharedPolicy, evaluate, gae, ppo_loss_and_grads, train  # noqa: E402


def _batch(rng, n_obs=7, n_act=4, B=16):
    mask = (rng.random((B, n_act)) > 0.3).astype(np.int8)
    mask[np.arange(B), rng.integers(0, n_act, B)] = 1
    act = np.array([rng.choice(np.flatnonzero(m)) for m in mask])
    return {"obs": rng.normal(size=(B, n_obs)), "mask": mask, "act": act,
            "logp": np.log(rng.uniform(0.1, 0.6, B)), "adv": rng.normal(size=B),
            "ret": rng.normal(size=B)}


def test_ppo_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    pol = SharedPolicy(7, 4, seed=1, hidden=8)
    b = _batch(rng)
    _, grads, _ = ppo_loss_and_grads(pol, b, clip=0.2, ent_coef=0.05, vf_coef=0.5)
    nets = {"a": pol.actor, "c": pol.critic}
    eps = 1e-6
    for key in ("a_W1", "a_b2", "a_W3", "c_W1", "c_W3", "c_b3"):
        net, name = nets[key[0]], key[2:]
        P = net.p[name]
        for _ in range(4):
            i = tuple(rng.integers(0, s) for s in P.shape)
            old = P[i]
            P[i] = old + eps
            lp, _, _ = ppo_loss_and_grads(pol, b, 0.2, 0.05, 0.5)
            P[i] = old - eps
            lm, _, _ = ppo_loss_and_grads(pol, b, 0.2, 0.05, 0.5)
            P[i] = old
            num = (lp - lm) / (2 * eps)
            assert grads[key][i] == pytest.approx(num, rel=1e-4, abs=1e-8), key


def test_masked_actions_get_zero_probability():
    rng = np.random.default_rng(1)
    pol = SharedPolicy(7, 4, seed=2)
    mask = np.array([1, 0, 1, 0], dtype=np.int8)
    obs = rng.normal(size=7)
    assert pol.probs(obs, mask)[0][[1, 3]].sum() == 0.0
    for _ in range(200):
        a, _, _ = pol.act(obs, mask, rng)
        assert mask[a] == 1


def test_gae_with_terminal_reward_only():
    # gamma = lam = 1: every advantage is the final reward minus that decision's value
    adv = gae(np.array([0.0, 0.0, -2.0]), np.array([-1.0, -1.5, -1.8]), gamma=1.0, lam=1.0)
    assert adv == pytest.approx([-1.0, -0.5, -0.2])


def test_short_training_run_and_save_load(tmp_path):
    pol, curve = train(setting="B_d", updates=2, episodes_per_update=1, seed=0)
    assert len(curve) == 2 and np.isfinite(curve[-1]["mean_reward"])
    path = tmp_path / "pol.npz"
    pol.save(path)
    back = SharedPolicy.load(path)
    assert back.meta["setting"] == "B_d"
    r1 = evaluate(pol, levels=("normal",), m=1)
    r2 = evaluate(back, levels=("normal",), m=1)
    for r in (r1[0], r2[0]):
        r.pop("decision_us")
    assert r1 == r2
