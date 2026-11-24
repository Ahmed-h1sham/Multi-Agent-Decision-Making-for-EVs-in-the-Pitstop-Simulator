# EV Pitstop thesis: multi-agent extension

Handoff from claude.ai sessions on 14 Oct, 2 Nov and 3 Nov 2026. Read this first, then README.md.

## Context
- Ahmed's GIU bachelor thesis: "Multi-Agent Decision Making for EVs in the Pitstop Simulator".
  Supervisors: Dr. Amal Aboulhassan and Dr. Yomna Hassan.
- Predecessor: Youssef Osman's `ev-simulator` (one EV on a segmented trip, continue-or-charge).
  This package is a new shared-charger environment that reuses his energy and charging models.
  It is not a fork of his repo.
- Scope: urban area, several stations, finite chargers, FCFS queues, charging only (no tyres).
- Core comparison: A (independent choice) vs B (reservation board). Secondary: B_info
  (board shown, plain FCFS service) to separate the information effect from the priority effect.
  A_d / B_d add the defer action (wait where you are, decide again later).
- Dr. Amal agreed (Nov 2026) that the thesis does not have to extend Osman's code: reuse what
  is needed (energy.py) and cite it.
- `pitstop_mas/env.py` exposes the simulator as a PettingZoo AEC environment (agents = EVs);
  this is the interface for a learned policy.
- Metrics: w (queue wait), d (request to start of charging), p_tau (share of w > tau),
  C (completed / all requests), with stranded, abandoned and unfinished counted separately.
- Evaluation: pre-generated matched scenarios, paired differences, paired t-test.
- Writing thesis text: use Youssef Osman's thesis PDF as the model for style and structure.

## Commands
- `python run_demo.py`: tests, Erlang C check, 30 paired scenarios x 3 levels x 5 policies,
  defer report, PettingZoo check, sensitivity sweeps, PPO evaluation, writes `results/` and
  `results/pitstop_demo.html` (about 35 s)
- `python run_demo.py --m 10 --skip-tests`: quick run
- `python -m pytest -q tests`: 55 validation tests (core + PettingZoo + PPO), all must pass
- `python train_ppo.py [--setting B_d|A_d] [--seeds 0 1 2] [--updates 800]`: train and evaluate
  parameter-shared PPO; models in `results/ppo/`, picked up by `run_demo.py`

## Code map
- `pitstop_mas/energy.py`: energy use, charger power, SOC-band charging time (reused, corrected)
- `pitstop_mas/station.py`: chargers, queue, service rules, outages, `project_start`
- `pitstop_mas/sim.py`: event-driven simulation, per-request log (`Rec`); `run()` lets the
  policy decide, `advance()` / `act()` pause at each decision for an outside controller
- `pitstop_mas/policies.py`: policies A / B_info / B / A_d / B_d, `Policy.decide` (defer rule),
  `ReservationBoard` with its contract
- `pitstop_mas/env.py`: PettingZoo AEC environment (`PitstopEnv`, `env()`, `rule_action`)
- `pitstop_mas/learn.py`: parameter-shared PPO in NumPy (`train`, `evaluate`, `ppo_report`);
  training scenarios use base seed 1000 (TRAIN_SEED), evaluation uses 2027 (EVAL_SEED)
- `pitstop_mas/scenario.py`: scenario generator (all randomness lives here), measurement window
  (`Scenario.measured`), `toy_herding`
- `pitstop_mas/config.py`: fixed method values and seed sets
- `pitstop_mas/metrics.py`, `stats.py`, `experiment.py`: metrics, paired stats, paired runner
- `pitstop_mas/validation.py`: single M/M/c station vs Erlang C
- `pitstop_mas/replay_template.html`: demo page template; `run_demo.py` fills `/*__DATA__*/null`

## Rules to keep
- Randomness only in `scenario.generate` (SeedSequence per level and index). Simulator and
  policies stay deterministic; the scenario digest must not change when a policy runs.
- Every policy gets the same public station information; only the board may differ.
- Keep the fixed event order (`sim.EVENT_ORDER`); ties by request id.
- An empty denominator gives NaN (undefined), never 0. Report w and C together.
- Every new rule gets a test in `tests/test_core.py` (environment rules in `tests/test_env.py`).
- The PettingZoo environment adds no behaviour: rule policies through it must give identical
  runs (`test_rule_policies_identical_through_pettingzoo`).
- Deferral time counts in d (from the original request) and uses up patience; judge deferral
  on d and C, never on w alone.

## Fixed method values (3 Nov 2026; reasons and sources in README "Method values")
- Warm-up 4 h (Welch's procedure, seed 777), measurement window: requests activated in
  [4 h, 10 h), 2 h drain, run ends at 12 h. Only measured requests count in the metrics.
- tau = 15 min (also reported at 5, 10, 30). Patience U(15, 60) min (Hanni et al. 2024).
- Demand: normal rho 0.6, high rho 0.9, reduced rho 0.6 with 3 of 10 chargers out for the
  whole run (Rempel et al. 2023: 27.5 % of DCFC connectors not working).
- Seeds: DEV 2026 (development), WELCH 777, TRAIN 1000 (PPO), EVAL 2027 (final results).
- Unchanged: choice rule (reachable station with earliest projected start; ties by shorter
  trip, then station id); board contract (book at decision, visible at once, 10 min arrival
  tolerance, abandoning frees the place, B serves valid bookings in booking order, no charger
  idle while a car waits); defer rule (defer if projected queue wait >= 5 min, wait 5 min, at
  most 3 times, no booking while deferring); environment reward -d/60 h, 2 h give-up penalty.
- Model: 12 x 12 km, 5 stations / 10 chargers, 30 km/h, road = 1.3 x straight line,
  SOC 8-35 % -> 80 %.
- Do not change these values after looking at EVAL results. A new value means a new dev
  round on DEV_SEED and a rerun on EVAL_SEED.

## Final results (seed 2027, 30 scenarios per level, steady state)
- Completion A -> B: normal 96.9 -> 99.2 %, high 88.3 -> 94.6 %, reduced 85.5 -> 91.3 %;
  give-ups roughly halved under load. This is the board's main gain.
- Delay depends on load. Normal: B shortens w (-1.23 min) and d (-0.73). High and reduced:
  w unchanged; d over requests served under both (`d_common`) is +1.34 / +1.62 min longer,
  from longer trips (high: A 7.5, B_info 8.1, B 8.9 min).
- Plain mean d is survivor-biased (drivers who give up drop out), so always report
  `d_common` / `w_common` next to it.
- Information vs priority (high): B_info gets most of the completion gain with no delay cost;
  booking priority adds +1.7 pp completion and +1.15 min d_common (p = 0.0007).
- Sensitivity: completion gain at every rho and patience range; delay crossover between
  rho 0.7 and 0.8; with tau = 5 min the board gives more long waits under load, with
  tau = 30 fewer.
- Defer: B_d nearly halves w under load, d not significantly changed, completion -1.1 pp at
  high (p = 0.015). Judge on d and C.
- PPO (retrained, mean of 3 seeds): with the board it ends up at rule B (no deferring); without
  the board it raises completion over A_d by 1.5 pp but stays well below B.
- Development-seed results without a warm-up (board shortened d at high demand) are
  superseded; the empty start flattered B.

## Open items
- Hold rule (keep a charger for a booked car that has not arrived) is not implemented.
- A board rule that weighs the extra trip against the shorter queue (source of the delay cost).
- Learned policy: untried centralised critic (MAPPO), reward without the 2 h give-up penalty.
- Verify the journal details of Rempel et al. (DOI 10.1177/00187208231215242) before citing.
- No Minecraft visual layer (decided Nov 2026); the HTML replay page is the visual demo.
- Thesis background chapter ("Why Extend Pitstop") still claims the Gymnasium interface and the
  Q-learning / PPO code carry over from Osman; only the energy and charging models do.
  Replacement text and the PettingZoo citation (terry2021pettingzoo) were given on 3 Nov 2026.
- Thesis text: Chapter 2 revisions from 14 Oct are in the claude.ai project doc
  `claude/literature_review.tex` and not yet merged into the thesis.
- Method values and their sources are in the claude.ai project doc `claude/method_values.md`.
