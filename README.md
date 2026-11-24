# Pitstop multi-agent extension: demo build

Multi-Agent Decision Making for EVs in the Pitstop Simulator (GIU bachelor thesis).
Several EVs choose between urban charging stations with finite chargers and FCFS queues.
The demo compares **A, independent choice**, with **B, a shared reservation board**, on the
same pre-generated scenarios, and adds **B-info** (board shown, plain FCFS service) to
separate the information effect from the priority effect. **A_d** and **B_d** add the
**defer action**: an EV may wait where it is and decide again later.

The simulator is also available as a **PettingZoo multi-agent environment** (`pitstop_mas/env.py`):
each EV is an agent that picks a station or defers, on the same simulation core. A
**parameter-shared PPO** policy (`pitstop_mas/learn.py`, `train_ppo.py`) is trained on it and
compared with the rules.

This is a new environment that reuses the energy and charging models of the earlier
Pitstop EV adaptation (Osman, 2026). The old environment has no stations, chargers or
queues, so the segment loop was not edited.

## Structure

| Layer | File | Role |
|---|---|---|
| Simulation | `pitstop_mas/energy.py` | Energy use, charger power, SOC-band charging time (reused and corrected) |
| Simulation | `pitstop_mas/station.py` | Station, chargers, queue, service rule, outages, `project_start` |
| Simulation | `pitstop_mas/sim.py` | Event-driven multi-EV simulation, per-request log; runs on its own (`run`) or pauses at each decision (`advance` / `act`) |
| Agents | `pitstop_mas/policies.py` | Policies A, B-info, B, A_d, B_d, the defer rule and the reservation board with its contract |
| Agent interface | `pitstop_mas/env.py` | PettingZoo AEC environment on the same core (agents = EVs) |
| Learning | `pitstop_mas/learn.py`, `train_ppo.py` | Parameter-shared PPO in NumPy: training, evaluation, comparison with the rules |
| Scenarios | `pitstop_mas/scenario.py` | Pre-generated scenarios (normal, high, reduced capacity), measurement window, toy herding case |
| Evaluation | `pitstop_mas/metrics.py`, `stats.py`, `experiment.py` | w, d, p_tau, C, utilisation, outcome counts, deferral; paired runner; served-under-both comparison; sensitivity sweeps |
| Validation | `pitstop_mas/validation.py`, `tests/test_core.py`, `tests/test_env.py`, `tests/test_learn.py` | Erlang C check and 55 tests |
| Presentation | `pitstop_mas/replay_template.html` | Demo page, filled by `run_demo.py` |

## Environment setup

```
python -m venv venv
venv\Scripts\activate            # Windows (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

Python 3.10 or newer.

## Running

```
python run_demo.py               # tests, Erlang C, paired experiment, sensitivity, PPO evaluation, demo page (~35 s)
python run_demo.py --m 10        # quicker
python run_demo.py --skip-tests  # skip pytest
python -m pytest -q tests        # tests only
python train_ppo.py              # train PPO for B_d (3 seeds, ~3.5 min each), evaluate, compare
python train_ppo.py --setting A_d
```

`run_demo.py` evaluates every model saved in `results/ppo/` and adds it to the demo page.

Outputs in `results/`:

| File | Content |
|---|---|
| `runs.csv` | One row per level x scenario x policy (all metrics and outcome counts) |
| `summary.json` | Paired differences, 95% CIs, paired and Welch p-values, per-scenario differences, served-under-both delay and wait |
| `sensitivity.json` | A vs B for rho 0.5-0.95, three patience ranges, and tau = 5, 10, 15, 30 min |
| `validation.json` | Erlang C table and toy case |
| `replay.json` | Per-request log of the replayed scenario under A and B |
| `env_check.json` | PettingZoo check: every policy through the environment vs the direct run |
| `ppo/` | Trained PPO models (3 seeds per setting), learning curves, reports |
| `pitstop_demo.html` | The demo page (open in a browser) |

## Method values (fixed 3 Nov 2026)

These values were fixed before the final run and are not changed after seeing its results.
Development used base seed 2026; the final results below use a separate seed set (2027),
as the proposal requires.

| Value | Fixed at | Reason | Source |
|---|---|---|---|
| Warm-up | 4 h, discarded | Welch's procedure on 150 replications x 12 h (seed 777), mean queue wait in 15-min bins, moving average. High demand is within 5 % of its plateau from 3.5 h; normal and reduced capacity from 2-3 h. 4 h covers all levels and both policies. Measuring from 0 h understated waiting by 25-74 % and favoured B. | Law (2015), Welch's procedure |
| Measurement window | Requests activated in [4 h, 10 h); the run continues 2 h more (12 h total) | 6 h gives about 47 (normal) to 71 (high) measured requests per run. The 2 h drain lets measured requests finish: 4 unfinished requests in 450 runs. | |
| Waiting threshold tau | 15 min | Upper edge of the shortest acceptable-wait category (5 to under 15 min) in a driver survey, and the lower end of the patience range. Reported next to 5, 10 and 30 min. | Hanni et al. (2024) |
| Patience | Uniform 15-60 min of queueing | The 15 to under 60 min acceptable-wait category of the same survey. Cross-check: an impatience factor of 0.6 x charging time with an average paid DC fast-charging session of 42 min gives about 25 min, inside the range. Sweep: 5-30 and 30-90 min. | Hanni et al. (2024); Chattopadhyay & Kar; DOE FOTW #1319 |
| Normal demand | rho = 0.6 | Below the knee of the Erlang C waiting curve: queues are short and rare. Realised utilisation 60-61 %. | Erlang C (validated in `validation.py`) |
| High demand | rho = 0.9 | Above the knee: waiting grows sharply and drivers start to give up. Realised utilisation 82-87 %. Real networks average far less over a day, so the levels stand for busy periods, not daily averages. Sweep: rho 0.5-0.95. | Erlang C; EVAdoption utilisation report |
| Reduced capacity | rho = 0.6, 3 of 10 chargers out for the whole run | A survey of 657 public DC fast-charging connectors found 27.5 % not working; 3 of 10 is the nearest whole number. The survey found chargers broken at the time of visit, not briefly down, so the outage lasts the whole run. | Rempel et al. (2023) |
| Seeds | Development 2026, Welch 777, PPO training 1000, final evaluation 2027 | Values chosen on one set and reported on another, so results cannot be tuned to the evaluation scenarios. | |

Unchanged from the demo: booking tolerance 10 min; defer rule (defer if the projected
queue wait at the best station is at least 5 min, wait 5 min, at most 3 times).

Other model values:

| Parameter | Value | Where |
|---|---|---|
| Area | 12 x 12 km, road distance = 1.3 x straight line, 30 km/h | `config.py` |
| Stations | Downtown DC x3, North DC x2, East ultra x1, South DC x2, West DC x2 (10 chargers) | `config.py` |
| Charger power | DC fast 50 kW, ultra-fast 150 kW | `energy.py` (from `env/gym_ev_env.py`) |
| Energy use | 0.18 kWh/km x (1 + (1 - SOC) x 0.05) | `energy.py` (from `env/gym_ev_env.py`) |
| Batteries | Model 3 75, Leaf 62, i4 83, Ioniq 77, Mach-E 88 kWh | `energy.py` (from `streamlit_app_ev.py`) |
| Demand | Poisson; 55 % of requests around Downtown (sd 1.8 km), rest uniform | `config.py` |
| Initial / target SOC | 8-35 % / 80 % | `config.py` |
| Travel noise | log-normal, sd 0.15, owned by the scenario | `config.py` |

rho is nominal: lambda divided by the sum of charger service rates, ignoring travel energy.
Realised utilisation (busy charger-minutes / available charger-minutes in the measurement
window) is in `runs.csv`.

## PettingZoo environment

```python
from pitstop_mas.env import env

e = env(setting="B_d", level="high")      # setting = A, B_info, B, A_d or B_d
e.reset(seed=0)                           # seed = scenario index
for agent in e.agent_iter():
    obs, reward, done, cut, info = e.last()
    if done or cut:
        action = None                     # a finished EV only takes None
    else:
        action = my_policy(obs)           # obs["action_mask"] marks the legal actions
    e.step(action)
```

| Part | Definition |
|---|---|
| Agents | One per charging request (`ev_<id>`). It joins when its need appears and leaves once it starts charging, gives up, or the run ends. |
| Turn order | PettingZoo AEC: one EV at a time, in the simulator's event order. |
| Observation | Per station: reachable, travel time, projected start, queue length, idle chargers, EVs on their way (board settings only), charger power. Own: SOC, time since request, deferrals used, clock. Plus an action mask. |
| Action | `Discrete(K + 1)`: a station, or defer. Unreachable stations and (outside A_d / B_d) defer are masked. An illegal action falls back to the rule's station and is flagged in `info`. |
| Reward | Paid once, when access is settled: `-d / 60` (hours); giving up adds a 2 h penalty; truncated at the end of the run. |
| Setting | A, B_info, B, A_d, B_d fix what EVs see and how stations serve; only the decision comes from the agent. |

`pitstop_mas.env.rule_action(e)` returns the action a rule-based policy would take, so the
thesis policies can be run through the environment. `tests/test_env.py` checks that this
gives exactly the same runs as the direct simulation, and runs PettingZoo's own `api_test`.

## Defer action

An EV that would queue at least `defer_step` minutes (5) at its best station anyway waits
where it is instead and decides again after `defer_step` minutes, at most `max_defer` (3)
times. While deferring it books nothing; once it commits it books as usual. Its delay d
counts from the original request, and deferral time uses up its patience, so deferral
cannot hide waiting or buy extra queueing time.

## Learned policy: parameter-shared PPO

Every EV uses the same actor-critic network (two hidden layers of 64, tanh) and decides
from its own observation. Its reward is its own -d / 60 (hours), with a 2 h penalty for
giving up: the same objective as the rules plus a failure penalty. Transitions are
collected per EV, advantages use GAE along each EV's own decisions (gamma = 1,
lambda = 0.95), and the update is clipped PPO (clip 0.2, Adam, lr 1e-3, entropy 0.003,
4 epochs, minibatch 256, gradient norm 0.5).

It is written in NumPy (`learn.py`) because SB3-style single-agent training assumes that
an agent's reward follows its own action. Here many EVs act in turn and each EV's reward
arrives only after others have acted. The gradients are checked against finite
differences in `tests/test_learn.py`.

- **Training:** 800 updates x 8 episodes = 6,400 episodes per seed, about 3.5 min on a
  laptop CPU. Scenarios come from base seed 1000, demand levels in turn, and are never
  evaluation scenarios. Hyperparameters were chosen on training scenarios only. The
  models were retrained after the method values were fixed.
- **Evaluation:** the greedy policy runs on the same 30 scenarios per level as the rules
  (base seed 2027). Comparisons are paired. Results are the mean over 3 training seeds.

| Level | Rule A_d | PPO (A_d) | Rule B_d | PPO (B_d) | Rule B |
|---|---|---|---|---|---|
| Normal, d (min) | 8.70 | 8.75 | 8.13 | 8.07 | 8.04 |
| High, d (min) | 13.97 | 14.04 | 14.90 | 15.12 | 15.39 |
| Reduced, d (min) | 15.02 | 16.04 | 16.34 | 17.36 | 16.84 |
| High, gave up (per run) | 8.63 | 7.56 | 5.07 | 4.20 | 4.23 |
| High, completion | 88.5 % | 90.0 % | 93.5 % | 94.7 % | 94.6 % |
| Reduced, completion | 85.5 % | 87.0 % | 90.4 % | 91.2 % | 91.3 % |
| High, EVs deferring | 7 % | 0 % | 27 % | 0 % | 0 % |

12. **With the board, learning ends up at rule B.** PPO (B_d) never defers and matches B on
    d and completion at every level (high: d -0.27 [-0.67, +0.14] min, completion +0.1 pp).
    Against B_d it completes more at high demand (+1.2 pp, p = 0.0004; 0.87 fewer give-ups
    per run) with no significant change in d, but d is 1.0 min longer in reduced capacity
    (p = 0.008).
13. **Without the board, learning trades delay for completion.** PPO (A_d) raises completion
    over A_d by 1.5 pp at high demand and reduced capacity (p <= 0.02; about 1 fewer give-up
    per run). d is unchanged at high demand (+0.07, n.s.) but 1.0 min longer in reduced
    capacity (p = 0.002). Its reward penalises giving up; the rule only minimises the
    projected start.
14. **Learning does not replace the information.** PPO (A_d) stays below rule B on completion
    at every level (-2.2, -4.6, -4.3 pp; p < 0.0001).

Per-seed results are in `results/ppo/ppo_report_<setting>.json`.

## Random seeds and reproducibility

All randomness is drawn once per scenario in `scenario.generate` from
`SeedSequence([base_seed, level, index])`. Policies and the simulator use no random
numbers, so every policy sees exactly the same requests, locations, SOCs, patience,
travel noise and outages. The scenario digest (which includes the measurement window) is
checked before and after each policy run. Simultaneous events follow a fixed order
(`sim.EVENT_ORDER`), ties by request id. The seed sets are `DEV_SEED`, `WELCH_SEED`,
`TRAIN_SEED` and `EVAL_SEED` in `config.py`. Two full runs give identical results apart
from the measured decision time.

## Validation

- One M/M/c station built from the same `Station` class matches Erlang C for c = 1, 2, 4
  and rho = 0.5, 0.7, 0.9 (formula inside the 95% batch-means interval in all 9 cases).
- Tests: capacity never exceeded (outages included), FCFS order kept, energy balance,
  faster charger gives same energy in less time (old issue 1), booking order, tolerance
  and expiry, valid bookings served before lapsed ones, outage removes a charger,
  scenarios reproducible and policy-independent (old issue 2), repeatable runs, empty
  denominator reported as undefined, and the toy herding case: three EVs, two stations;
  independent choice sends all three to A, the board sends them A, B, A, exactly the
  background-chapter figure.
- Method-value tests: only requests after the warm-up are measured, the reduced-capacity
  outage lasts the whole run, patience stays in the fixed range.
- Defer tests: just-in-time departure keeps the same start of charging, deferral counts in
  d from the original request, a deferring EV books only when it commits, deferral uses up
  patience, and with no deferrals allowed A_d = A and B_d = B.
- PettingZoo tests: PettingZoo's `api_test`, all five policies identical through the
  environment, reward = -d, random masked agents keep capacity and limits, masks, illegal
  actions, observation space.

## Results (final run: seed 2027, 30 scenarios per level, steady state)

Mean over scenarios, A -> B, with the paired difference and 95% CI. "d, served under both"
is the delay over the requests that start charging under both policies (96 % of requests at
normal demand, 85 % at high demand, 81 % in reduced capacity).

| Level | Completion C | Gave up (per run) | Mean wait w (min) | Wait > 15 min | Access delay d (min) | d, served under both |
|---|---|---|---|---|---|---|
| Normal | 96.9 % -> 99.2 % | 1.47 -> 0.37 | 2.51 -> 1.27 (-1.23 [-1.58, -0.89]) | 6.6 % -> 2.2 % | 8.77 -> 8.04 (-0.73 [-1.08, -0.39]) | -0.91 [-1.25, -0.57] |
| High | 88.3 % -> 94.6 % | 8.73 -> 4.23 | 6.20 -> 6.19 (-0.01 [-0.81, +0.78]) | 16.3 % -> 15.4 % | 13.73 -> 15.39 (+1.66 [+0.66, +2.66]) | +1.34 [+0.34, +2.34] |
| Reduced | 85.5 % -> 91.3 % | 7.47 -> 4.50 | 6.25 -> 6.83 (+0.58 [-0.66, +1.82]) | 17.2 % -> 18.6 % | 14.87 -> 16.84 (+1.97 [+0.56, +3.37]) | +1.62 [+0.23, +3.00] |

Points to discuss:

1. **The board's main gain is completion.** It raises completion by 2.3 / 6.3 / 5.8 pp and
   halves the drivers who give up at high demand and in reduced capacity (p < 1e-8).
   Utilisation rises with it (high: 81.7 % -> 86.6 %).
2. **Its effect on delay depends on load.** At normal demand the board shortens wait and
   delay. Under heavy load queue wait does not change, and delay is 1.3-1.6 min longer for
   the drivers served under both policies. The extra delay is driving: the board sends
   drivers to farther stations (mean trip at high demand: A 7.5, B-info 8.1, B 8.9 min).
3. **Survivor bias.** Plain mean delay over served drivers favours the policy that loses
   more drivers, because the drivers who give up are those who would have waited longest.
   This is why d is also compared over the requests served under both, and why w and C are
   always reported together.
4. **Information versus priority.** At high demand B-info (board shown, plain FCFS) gets
   most of the completion gain (+4.6 pp) with no change in delay for the drivers served
   under both (-0.03 [-0.86, +0.80]). The booking priority adds +1.7 pp completion and 1.2
   fewer give-ups per run, and it adds the extra delay (+1.15 [+0.53, +1.76] min,
   p = 0.0007). In reduced capacity B-info already lengthens delay (+1.38 [-0.01, +2.76],
   p = 0.051) and the priority adds little.
5. **Herding shows up in the estimates.** At high demand A's projected start is 5.2 min too
   early on average; B's is 0.9 min. Chargers idle while another station had a queue drop
   from 360 to 132 charger-minutes per run.
6. **Pairing matters.** For mean wait at normal demand the paired 95% CI is +/-0.34 min;
   treating the same runs as independent groups gives +/-0.57 min (+/-0.80 vs +/-1.38 at
   high demand).
7. **Decision time** is about 18-20 us (A), 22-26 us (B) and 38-40 us (B_d) per decision.
8. **The empty start mattered.** On the development seeds without a warm-up the board also
   shortened delay at high demand. With the 4 h warm-up this reverses. The steady-state
   result is the one reported.

### Sensitivity (`results/sensitivity.json`)

| Sweep | Completion A -> B | d served under both, B - A (min) |
|---|---|---|
| rho 0.5 | 97.6 % -> 99.3 % | -0.68 [-0.93, -0.43] |
| rho 0.6 (fixed) | 96.1 % -> 98.8 % | -0.41 [-0.94, +0.11] |
| rho 0.7 | 93.8 % -> 97.9 % | -0.39 [-1.01, +0.24] |
| rho 0.8 | 91.4 % -> 96.5 % | +0.23 [-0.58, +1.04] |
| rho 0.9 (fixed) | 88.0 % -> 93.3 % | +1.31 [+0.27, +2.36] |
| rho 0.95 | 86.3 % -> 91.3 % | +2.03 [+1.08, +2.98] |
| Patience 5-30 min, high demand | 82.2 % -> 88.6 % | +1.32 [+0.79, +1.85] |
| Patience 15-60 min (fixed), high demand | 88.3 % -> 94.6 % | +1.34 [+0.34, +2.34] |
| Patience 30-90 min, high demand | 92.9 % -> 97.8 % | -0.50 [-1.96, +0.95] |

The rho sweep uses its own scenarios (no outages), so its rho 0.6 and 0.9 rows differ
slightly from the main table.

| Share of waits over tau | Normal, A -> B | High, A -> B | Reduced, A -> B |
|---|---|---|---|
| 5 min | 13.5 % -> 8.1 % | 30.5 % -> 38.0 % | 29.6 % -> 37.8 % |
| 10 min | 9.1 % -> 3.7 % | 22.4 % -> 24.7 % | 21.3 % -> 26.8 % |
| 15 min (fixed) | 6.6 % -> 2.2 % | 16.3 % -> 15.4 % | 17.2 % -> 18.6 % |
| 30 min | 1.7 % -> 0.4 % | 6.1 % -> 3.0 % | 6.9 % -> 4.0 % |

9. **The completion gain holds everywhere**: every rho and every patience range.
10. **The delay result has a crossover** between rho 0.7 and 0.8 (realised utilisation about
    72-78 %): below it the board shortens delay, above it the board lengthens it. Very
    patient drivers (30-90 min) remove the delay cost.
11. **tau changes the picture under load.** The board spreads waiting: more waits over 5 min,
    fewer over 30 min. A single tau would hide this, so the thesis reports the whole row.

### Defer action (A -> A_d, B -> B_d)

| Level | EVs deferring | Mean wait w (min) | Access delay d (min) | Completion C |
|---|---|---|---|---|
| Normal, A -> A_d | 3 % | 2.51 -> 2.30 | 8.77 -> 8.70 (-0.08 [-0.26, +0.11]) | 96.9 % -> 96.8 % |
| Normal, B -> B_d | 5 % | 1.27 -> 1.05 | 8.04 -> 8.13 (+0.09 [-0.04, +0.21]) | 99.2 % -> 99.3 % |
| High, A -> A_d | 7 % | 6.20 -> 5.96 | 13.73 -> 13.97 (+0.25 [-0.30, +0.79]) | 88.3 % -> 88.5 % |
| High, B -> B_d | 27 % | 6.19 -> 3.41 | 15.39 -> 14.90 (-0.49 [-1.00, +0.03]) | 94.6 % -> 93.5 % |
| Reduced, A -> A_d | 9 % | 6.25 -> 5.72 | 14.87 -> 15.02 (+0.14 [-0.36, +0.64]) | 85.5 % -> 85.5 % |
| Reduced, B -> B_d | 30 % | 6.83 -> 3.51 | 16.84 -> 16.34 (-0.50 [-1.20, +0.19]) | 91.3 % -> 90.4 % |

- **Deferral moves waiting; it does not remove it.** B_d nearly halves queue wait under load
  and cuts waits over 15 min from 15 % to 7 % at high demand, but d does not change
  significantly and completion drops a little (-1.1 pp at high demand, p = 0.015), because
  deferral time uses up patience. This is why d and C, not w, are the measures for deferral.
- **Without the board, EVs rarely defer** (3-9 %): they cannot see the cars already on their
  way, so they see few queues coming.

All scenarios are synthetic.

## Not built yet

- The hold rule (keeping a charger free for a booked vehicle that has not arrived).
- A centralised critic (MAPPO) and other reward definitions for the learned policy (the 2 h
  give-up penalty is untested for sensitivity).
- A board rule that weighs the extra trip against the shorter queue (under heavy load the
  delay cost comes from longer trips).

## Sources for the method values

- Hanni et al. (2024), acceptable waiting time at public charging, *Sustainability* 16(6), 2536. https://www.mdpi.com/2071-1050/16/6/2536
- U.S. DOE, Fact of the Week #1319 (4 Dec 2023): paid DC fast-charging sessions average 42 min. https://www.energy.gov/cmei/vehicles/articles/fotw-1319-december-4-2023-ev-charging-paid-dc-fast-charging-stations-average
- Rempel et al. (2023), reliability of public DC fast chargers (27.5 % of 657 connectors not working). https://arxiv.org/abs/2203.16372, DOI 10.1177/00187208231215242
- Chattopadhyay & Kar, impatience factor of 0.6 x charging time. https://arxiv.org/html/2403.06223v1
- EVAdoption, Electrify America California utilisation report (Q3 2021). https://evadoption.com/new-report-electrify-america-california-corridor-urban-q3-2021-ev-charging-utilization
- Law, A. M. (2015), *Simulation Modeling and Analysis*, 5th ed., McGraw-Hill: Welch's procedure for the warm-up period.
