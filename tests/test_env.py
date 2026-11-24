"""Tests for the PettingZoo interface (pitstop_mas/env.py)."""
import numpy as np
import pytest

pytest.importorskip("pettingzoo")

from pettingzoo.test import api_test  # noqa: E402

from pitstop_mas.env import PitstopEnv, env, rule_action  # noqa: E402
from pitstop_mas.metrics import run_metrics  # noqa: E402
from pitstop_mas.policies import POLICIES  # noqa: E402
from pitstop_mas.scenario import generate  # noqa: E402
from pitstop_mas.sim import run  # noqa: E402

LEVELS = ("normal", "high", "reduced")


def _play(e, choose):
    """Run one episode; returns {agent: final cumulative reward}."""
    paid = {}
    for agent in e.agent_iter():
        _obs, reward, term, trunc, _info = e.last()
        if term or trunc:
            paid[agent] = reward
            e.step(None)
        else:
            e.step(choose(e))
    return paid


@pytest.mark.parametrize("setting", ["A", "B_d"])
def test_pettingzoo_api(setting):
    api_test(env(setting=setting, level="normal"), num_cycles=300)


@pytest.mark.parametrize("setting", list(POLICIES))
def test_rule_policies_identical_through_pettingzoo(setting):
    # PettingZoo is only an interface: the same decisions give the same simulation.
    for level in LEVELS:
        for j in range(2):
            e = PitstopEnv(setting=setting, level=level)
            e.reset(options={"index": j})
            _play(e, rule_action)
            direct = run_metrics(run(generate(level, j), POLICIES[setting]))
            via_env = run_metrics(e.sim)
            for m in (direct, via_env):
                m.pop("decision_us")
            assert via_env == direct


def test_reward_is_minus_access_delay():
    e = PitstopEnv(setting="B", level="high")
    e.reset(options={"index": 0})
    paid = _play(e, rule_action)
    for agent, reward in paid.items():
        rec = e.sim.recs[int(agent[3:])]
        if rec.t_start is not None:
            assert reward == pytest.approx(-rec.d / 60.0)
        elif rec.outcome == "abandoned":
            assert reward == pytest.approx(-((rec.t_leave - rec.t_need) + e.fail_penalty) / 60.0)


def test_random_agents_respect_masks_and_capacity():
    rng = np.random.default_rng(0)
    for setting in ("A_d", "B_d"):
        e = PitstopEnv(setting=setting, level="high")
        e.reset(options={"index": 3})

        def masked_random(env_):
            mask = env_.observe(env_.agent_selection)["action_mask"]
            return int(rng.choice(np.flatnonzero(mask)))

        _play(e, masked_random)
        assert e.agents == []
        for st in e.sim.stations:
            assert st.violations == [] and st.max_busy <= st.spec.n_chargers
        for rec in e.sim.recs.values():
            assert rec.n_defer <= e.cfg.max_defer
            assert rec.outcome != "open"


def test_defer_only_where_allowed():
    for setting, allowed in (("A", 0), ("B", 0), ("A_d", 1), ("B_d", 1)):
        e = PitstopEnv(setting=setting, level="normal")
        e.reset(options={"index": 0})
        mask = e.observe(e.agent_selection)["action_mask"]
        assert mask[e.DEFER] == allowed


def test_illegal_action_falls_back_and_is_flagged():
    e = PitstopEnv(setting="A", level="normal")
    e.reset(options={"index": 0})
    agent = e.agent_selection
    rid = int(agent[3:])
    e.step(e.DEFER)                       # defer is masked out under setting A
    rec = e.sim.recs[rid]
    assert rec.n_defer == 0 and rec.sid is not None
    if agent in e.infos:                  # still listed until its access is settled
        assert e.infos[agent].get("illegal_action")


def test_observation_matches_space():
    e = PitstopEnv(setting="B_d", level="high")
    e.reset(options={"index": 1})
    space = e.observation_space("ev_0")
    for _ in range(80):
        if not e.agents:
            break
        agent = e.agent_selection
        if e.terminations[agent] or e.truncations[agent]:
            e.step(None)
            continue
        assert space.contains(e.observe(agent))
        e.step(rule_action(e))
