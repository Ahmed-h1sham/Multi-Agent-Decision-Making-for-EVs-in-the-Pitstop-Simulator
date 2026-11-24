"""PettingZoo multi-agent environment on the same simulation core.

The simulator is sim.Simulation. This module adds nothing to the physics: it pauses the
simulation each time an EV has to decide, hands that EV its observation, applies its
action and runs on to the next decision. Rule-based and learned policies therefore face
exactly the same stations, queues, energy model and scenarios (tests/test_env.py checks
that the rule-based policies give identical results through this interface).

Agents      One agent per charging request ("ev_<request id>"). An agent appears when its
            charging need appears and leaves once its access to charging is settled.
API         PettingZoo AEC: EVs act one at a time, in the simulator's event order.
Setting     `setting` is a policy key from policies.POLICIES. It fixes what the EVs see
            and how stations serve; only the decision itself comes from outside:
              A       public station information, FCFS service
              B_info  + reservation board, FCFS service
              B       + reservation board, valid bookings served first
              A_d/B_d as A/B, and the defer action is allowed
Action      Discrete(K + 1): station index 0..K-1, or K = defer (wait defer_step minutes
            and decide again). An illegal action (masked out) falls back to the station
            with the earliest projected start and is flagged in infos.
Observation dict(observation=Box(0, 10, (7K + 4,)), action_mask=Box(0, 1, (K + 1,)))
            per station: reachable, travel time (h), projected start from now (h), queue
            length / 10, idle chargers / chargers, EVs on their way / 10 (board only),
            charger power / 150 kW; own: SOC, time since request (h), deferrals used /
            max, simulation time / t_end
Reward      Individual and paid once, when the agent's access is settled:
              starts charging   -d / 60          (d in hours)
              leaves the queue  -(time since request + fail_penalty) / 60
              run ends first    -(t_end - t_need) / 60   (truncation)
            These are defaults for a first learned policy, not settled choices.
"""
from __future__ import annotations

import math

import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from .config import DEFAULT_SIM, EVAL_SEED, SimConfig
from .energy import CHARGER_POWER_KW
from .policies import POLICIES
from .scenario import generate
from .sim import Simulation

PER_STATION = 7
OWN = 4
CAP = 10.0


def _name(rid: int) -> str:
    return f"ev_{rid}"


class PitstopEnv(AECEnv, EzPickle):
    metadata = {"name": "pitstop_mas_v0", "render_modes": ["ansi"], "is_parallelizable": False}

    def __init__(self, setting: str = "B", level: str = "high", base_seed: int = EVAL_SEED,
                 cfg: SimConfig = DEFAULT_SIM, max_agents: int = 200, fail_penalty: float = 120.0,
                 scenario_fn=None, render_mode: str | None = None):
        EzPickle.__init__(self, setting, level, base_seed, cfg, max_agents, fail_penalty,
                          scenario_fn, render_mode)
        super().__init__()
        if setting not in POLICIES:
            raise ValueError(f"unknown setting {setting!r}; use one of {list(POLICIES)}")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        self.setting, self.policy = setting, POLICIES[setting]
        self.level, self.base_seed, self.cfg = level, base_seed, cfg
        self.max_agents, self.fail_penalty = max_agents, fail_penalty
        self.render_mode = render_mode
        self._scenario_fn = scenario_fn or (lambda i: generate(level, i, base_seed, sim=cfg))

        probe = self._scenario_fn(0)
        self.station_ids = [s.sid for s in probe.stations]
        self.K = len(self.station_ids)
        self.DEFER = self.K
        self.possible_agents = [_name(i) for i in range(max_agents)]
        n_obs = PER_STATION * self.K + OWN
        self._obs_space = spaces.Dict({
            "observation": spaces.Box(0.0, CAP, (n_obs,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, (self.K + 1,), dtype=np.int8),
        })
        self._act_space = spaces.Discrete(self.K + 1)
        self.sim = None
        self.agents = []

    # ---- spaces ----------------------------------------------------------------
    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    # ---- episode ----------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Start an episode on scenario `options["index"]` (or `seed`, default 0)."""
        options = options or {}
        index = options.get("index", seed if seed is not None else 0)
        self.scenario = self._scenario_fn(index)
        if len(self.scenario.requests) > self.max_agents:
            raise ValueError(f"scenario has {len(self.scenario.requests)} requests; raise max_agents")
        if [s.sid for s in self.scenario.stations] != self.station_ids:
            raise ValueError("all scenarios of one environment must use the same stations")
        self.sim = Simulation(self.scenario, self.policy, self.cfg, external=True)
        self.agents = []
        self.rewards, self._cumulative_rewards = {}, {}
        self.terminations, self.truncations, self.infos = {}, {}, {}
        self._n_closed = 0
        self._decider = None
        self.agent_selection = None
        self._advance()

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            if action is not None:
                raise ValueError("a finished agent can only take the action None")
            self._remove(agent)
            self._clear_rewards()          # as PettingZoo's _was_dead_step does
            self._select()
            return
        if agent != self._decider:
            raise RuntimeError(f"{agent} is not the agent waiting for a decision")
        self._cumulative_rewards[agent] = 0.0
        self.rewards = {a: 0.0 for a in self.agents}
        rid, opts = self.sim.awaiting
        mask = self._mask(rid, opts)
        legal = action is not None and 0 <= int(action) <= self.K and mask[int(action)] == 1
        if legal:
            a = int(action)
            sid = None if a == self.DEFER else self.station_ids[a]
            self.infos[agent] = {}
        else:
            sid, _est, _seen = self.policy.choose(self.sim, opts)
            self.infos[agent] = {"illegal_action": True}
        self.sim.act(sid)
        self._advance()

    def observe(self, agent):
        if agent == self._decider and self.sim is not None and self.sim.awaiting is not None:
            return self._observation()
        n = self._obs_space["observation"].shape[0]
        return {"observation": np.zeros(n, dtype=np.float32),
                "action_mask": np.zeros(self.K + 1, dtype=np.int8)}

    def render(self):
        if self.render_mode is None:
            return None
        sim = self.sim
        lines = [f"t = {sim.now:6.1f} min   deciding: {self._decider}"]
        for st in sim.stations:
            out = sum(not c.in_service for c in st.chargers)
            enroute = len(sim.board.enroute.get(st.spec.sid, {})) if sim.board else 0
            lines.append(f"  {st.spec.name:9s} charging {st.busy}/{st.spec.n_chargers}"
                         f"  out {out}  queue {len(st.queue)}  on the way (board) {enroute}")
        return "\n".join(lines)

    def close(self):
        pass

    # ---- internals ---------------------------------------------------------------
    def _add(self, agent):
        self.agents.append(agent)
        self.rewards[agent] = 0.0
        self._cumulative_rewards[agent] = 0.0
        self.terminations[agent] = False
        self.truncations[agent] = False
        self.infos[agent] = {}

    def _remove(self, agent):
        self.agents.remove(agent)
        for d in (self.rewards, self._cumulative_rewards, self.terminations, self.truncations, self.infos):
            del d[agent]

    def _pay(self, agent, reward, info):
        self.rewards[agent] = self.rewards.get(agent, 0.0) + reward
        self._cumulative_rewards[agent] += reward
        self.infos[agent] = {**self.infos[agent], **info}

    def _advance(self):
        rid = self.sim.advance()
        self._collect_closed()
        if rid is None:
            self._decider = None
            for agent in self.agents:
                if not (self.terminations[agent] or self.truncations[agent]):
                    rec = self.sim.recs[int(agent[3:])]
                    self.truncations[agent] = True
                    self._pay(agent, -(self.cfg.t_end - rec.t_need) / 60.0, {"outcome": rec.outcome})
        else:
            agent = _name(rid)
            if agent not in self.terminations:
                self._add(agent)
            self._decider = agent
        self._select()

    def _collect_closed(self):
        for rid in self.sim.closed[self._n_closed:]:
            agent = _name(rid)
            if agent not in self.terminations or self.terminations[agent]:
                continue          # stranded before its first decision: never became an agent
            rec = self.sim.recs[rid]
            if rec.t_start is not None:
                reward = -rec.d / 60.0
            else:
                reward = -((rec.t_leave - rec.t_need) + self.fail_penalty) / 60.0
            self.terminations[agent] = True
            self._pay(agent, reward, {"outcome": rec.outcome, "d": rec.d, "w": rec.w})
        self._n_closed = len(self.sim.closed)

    def _select(self):
        finished = [a for a in self.agents if self.terminations[a] or self.truncations[a]]
        if finished:
            self.agent_selection = finished[0]
        elif self._decider is not None:
            self.agent_selection = self._decider

    def _mask(self, rid, opts):
        mask = np.zeros(self.K + 1, dtype=np.int8)
        reachable = {o.sid for o in opts}
        for k, sid in enumerate(self.station_ids):
            mask[k] = sid in reachable
        rec = self.sim.recs[rid]
        mask[self.DEFER] = self.policy.can_defer and rec.n_defer < self.cfg.max_defer
        return mask

    def _observation(self):
        sim, cfg = self.sim, self.cfg
        rid, opts = sim.awaiting
        r, rec = sim.req[rid], sim.recs[rid]
        seen = self.policy.estimates(sim, opts)
        by_sid = {o.sid: o for o in opts}
        feats = []
        for st in sim.stations:
            s, o = st.spec, by_sid.get(st.spec.sid)
            start_in = seen.get(s.sid, math.inf) - sim.now
            enroute = len(sim.board.enroute.get(s.sid, {})) if sim.board is not None else 0
            feats += [
                1.0 if o else 0.0,
                o.travel_est / 60.0 if o else 0.0,
                start_in / 60.0 if (o and math.isfinite(start_in)) else CAP,
                len(st.queue) / 10.0,
                st.idle_available() / s.n_chargers,
                enroute / 10.0,
                CHARGER_POWER_KW[s.charger_type] / 150.0,
            ]
        feats += [r.soc0, (sim.now - r.t_need) / 60.0, rec.n_defer / max(1, cfg.max_defer), sim.now / cfg.t_end]
        obs = np.clip(np.asarray(feats, dtype=np.float32), 0.0, CAP)
        return {"observation": obs, "action_mask": self._mask(rid, opts)}


def env(**kwargs):
    """PettingZoo-wrapped environment (order checks on); use PitstopEnv for the raw class."""
    return wrappers.OrderEnforcingWrapper(PitstopEnv(**kwargs))


def rule_action(e: PitstopEnv, rule=None) -> int:
    """Action that a rule-based policy would take for the agent now deciding.

    Used to drive the environment with the thesis policies (A, B_info, B, A_d, B_d).
    """
    rule = rule or e.policy
    rid, opts = e.sim.awaiting
    dec = rule.decide(e.sim, e.sim.recs[rid], opts)
    return e.DEFER if dec.sid is None else e.station_ids.index(dec.sid)
