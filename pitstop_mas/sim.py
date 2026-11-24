"""Event-driven multi-EV charging simulation (the new shared-charger environment).

This is a new environment, not an edit of the old segment loop: the earlier
environment has no stations, chargers or queues (reuse map, issue 7). It reuses the
energy and charging models through energy.py.

Event times are in minutes. Simultaneous events are processed in a fixed order so that
the comparison does not depend on an accident of event order (background, "Congestion
and Simultaneous Choices"); ties within one type go by insertion order, and requests are
inserted in request-id order.

Two ways to drive it, with the same mechanics:
  run()                  the policy's own rule makes every decision (rule-based policies)
  advance() / act()      the simulation pauses at each decision and an outside controller
                         decides (used by the PettingZoo environment in env.py)
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

from .config import DEFAULT_SIM, SimConfig
from .energy import CHARGER_POWER_KW, charge_minutes, drive_energy_kwh
from .policies import DEFER, GO, Decision, Option, Policy, ReservationBoard
from .station import BOOKING, Station, Waiting

EVENT_ORDER = {"outage_end": 0, "finish": 1, "outage_start": 2, "arrive": 3,
               "expire": 4, "renege": 5, "redecide": 6, "need": 7}


@dataclass
class Rec:
    """Per-request log (one row per charging request)."""
    rid: int
    t_need: float
    sid: int | None = None
    est_start: float | None = None   # projected start at decision time
    t_decide: float | None = None    # when the EV committed to a station
    t_arr_est: float | None = None
    t_arr: float | None = None
    t_start: float | None = None
    t_end: float | None = None
    t_leave: float | None = None
    soc_arr: float | None = None
    energy_kwh: float | None = None
    dur: float | None = None
    road_km: float | None = None
    travel: float | None = None
    decision_ns: int = 0
    n_options: int = 0
    n_defer: int = 0                 # times the EV deferred its choice
    deferred: float = 0.0            # minutes spent deferring
    bseq: int | None = None
    entitled: bool | None = None
    seen: dict | None = None         # projected start per reachable station at the last decision
    state: str = "pending"
    outcome: str = "open"

    @property
    def w(self):   # queue waiting at the station
        return None if self.t_start is None else self.t_start - self.t_arr

    @property
    def d(self):   # charging-access delay (includes deferral and travel)
        return None if self.t_start is None else self.t_start - self.t_need


class Simulation:
    def __init__(self, scenario, policy: Policy, cfg: SimConfig = DEFAULT_SIM, external: bool = False):
        self.sc, self.policy, self.cfg = scenario, policy, cfg
        self.external = external         # True: decisions come from act(), not from the policy rule
        self.stations = [Station(spec, policy.service_rule) for spec in scenario.stations]
        self.board = ReservationBoard(cfg.booking_tolerance) if policy.uses_board else None
        self.req = {r.rid: r for r in scenario.requests}
        self.recs = {r.rid: Rec(r.rid, r.t_need) for r in scenario.requests}
        self.now = 0.0
        self.idle_while_queued = 0.0     # charger-minutes idle while another station has a queue
        self.busy_minutes = 0.0          # charger-minutes charging (measurement window)
        self.capacity_minutes = 0.0      # charger-minutes in service (measurement window)
        self.closed = []                 # request ids whose access ended (started, left or stranded)
        self.awaiting = None             # (rid, options) while paused for an outside decision
        self.finished = False
        self._ev, self._seq, self._arr_seq, self._last_t = [], 0, 0, 0.0
        for r in scenario.requests:
            self._push(r.t_need, "need", r.rid)
        for o in scenario.outages:
            self._push(o.t_start, "outage_start", o)
            self._push(o.t_end, "outage_end", o)

    # ---- event loop ----------------------------------------------------------
    def _push(self, t, kind, data):
        heapq.heappush(self._ev, (t, EVENT_ORDER[kind], self._seq, kind, data))
        self._seq += 1

    def run(self) -> "Simulation":
        if self.external:
            raise RuntimeError("an externally controlled simulation is driven with advance() and act()")
        self.advance()
        return self

    def advance(self):
        """Process events until a decision is needed or the run ends.

        Returns the request id waiting for a decision, or None when the run is over.
        """
        t_end = self.cfg.t_end
        while self.awaiting is None and self._ev and self._ev[0][0] <= t_end:
            t, _, _, kind, data = heapq.heappop(self._ev)
            self._accumulate(t)
            self.now = t
            getattr(self, "_on_" + kind)(data)
        if self.awaiting is not None:
            return self.awaiting[0]
        self._finish_run()
        return None

    def act(self, sid: int | None) -> None:
        """Apply an outside decision for the paused vehicle: a station id, or None to defer."""
        if self.awaiting is None:
            raise RuntimeError("no vehicle is waiting for a decision")
        rid, opts = self.awaiting
        seen = self.policy.estimates(self, opts)
        if sid is None:
            dec = Decision(DEFER, None, None, seen)
        else:
            if sid not in seen:
                raise ValueError(f"station {sid} is not reachable for request {rid}")
            dec = Decision(GO, sid, seen[sid], seen)
        self.awaiting = None
        self._apply(rid, opts, dec)

    def _finish_run(self):
        if self.finished:
            return
        t_end = self.cfg.t_end
        self._accumulate(t_end)
        self.now = t_end
        for rec in self.recs.values():
            if rec.state != "done":
                rec.outcome = {"travel": "unfinished_travel", "queue": "unfinished_waiting",
                               "charging": "unfinished_charging",
                               "deferred": "unfinished_deferred"}.get(rec.state, "unfinished")
        self.finished = True

    def _accumulate(self, t):
        """Time-weighted counters over the measurement window [t_warm, t_meas_end)."""
        a, b = max(self._last_t, self.sc.t_warm), min(t, self.sc.t_meas_end)
        if b > a:
            dt = b - a
            self.busy_minutes += dt * sum(st.busy for st in self.stations)
            self.capacity_minutes += dt * sum(sum(c.in_service for c in st.chargers) for st in self.stations)
            if any(st.queue for st in self.stations):
                self.idle_while_queued += dt * sum(st.idle_available() for st in self.stations)
        self._last_t = max(self._last_t, t)

    # ---- decisions -----------------------------------------------------------
    def options_for(self, r) -> list:
        """Reachable stations: the vehicle keeps at least reserve_soc on arrival.

        A deferring EV waits where it is, so its options do not change while it waits.
        """
        opts = []
        for st in self.stations:
            s = st.spec
            road = math.hypot(r.x - s.x, r.y - s.y) * self.cfg.detour_factor
            soc_arr = r.soc0 - drive_energy_kwh(road, r.soc0) / r.capacity_kwh
            if soc_arr < self.cfg.reserve_soc:
                continue
            dur = charge_minutes(r.capacity_kwh, soc_arr, r.soc_target, CHARGER_POWER_KW[s.charger_type])
            opts.append(Option(s.sid, road, road * self.cfg.minutes_per_road_km, soc_arr, dur))
        return opts

    def _on_need(self, rid):
        self._decision_point(rid)

    def _on_redecide(self, rid):
        self._decision_point(rid)

    def _decision_point(self, rid):
        r, rec = self.req[rid], self.recs[rid]
        opts = self.options_for(r)
        rec.n_options = len(opts)
        if not opts:
            rec.state, rec.outcome = "done", "stranded"
            self.closed.append(rid)
            return
        if self.external:
            self.awaiting = (rid, opts)
            return
        t0 = time.perf_counter_ns()
        dec = self.policy.decide(self, rec, opts)
        rec.decision_ns += time.perf_counter_ns() - t0
        self._apply(rid, opts, dec)

    def _apply(self, rid, opts, dec: Decision):
        r, rec = self.req[rid], self.recs[rid]
        rec.seen = dec.seen
        if dec.action == DEFER:
            if not self.policy.can_defer or rec.n_defer >= self.cfg.max_defer:
                raise ValueError(f"request {rid} may not defer (policy {self.policy.key}, "
                                 f"{rec.n_defer} deferrals so far)")
            rec.n_defer += 1
            rec.deferred += self.cfg.defer_step
            rec.state = "deferred"
            self._push(self.now + self.cfg.defer_step, "redecide", rid)
            return
        sid = dec.sid
        opt = next(o for o in opts if o.sid == sid)
        rec.sid, rec.est_start, rec.road_km, rec.t_decide = sid, dec.est, opt.road_km, self.now
        rec.soc_arr, rec.dur = opt.soc_arr, opt.dur
        rec.energy_kwh = (r.soc_target - opt.soc_arr) * r.capacity_kwh
        rec.t_arr_est = self.now + opt.travel_est
        rec.travel = opt.travel_est * r.travel_noise
        rec.state = "travel"
        if self.board is not None:
            b = self.board.book(self.now, rid, sid, rec.t_arr_est, opt.dur)
            rec.bseq = b.bseq
            self._push(b.deadline, "expire", rid)
        self._push(self.now + rec.travel, "arrive", rid)

    # ---- station events --------------------------------------------------------
    def _on_arrive(self, rid):
        rec, st = self.recs[rid], self.stations[self.recs[rid].sid]
        rec.t_arr, rec.state = self.now, "queue"
        entitled = False
        if self.board is not None:
            entitled = self.board.arrived(self.now, rid) is not None
            rec.entitled = entitled
        self._arr_seq += 1
        key = (0, rec.bseq, 0) if (st.rule == BOOKING and entitled) else (1, self.now, self._arr_seq)
        st.join(Waiting(key, rid, self.now, rec.dur))
        # Time spent deferring uses up patience, so deferral cannot buy extra queueing time.
        patience_left = max(0.0, self.req[rid].patience - rec.deferred)
        self._push(self.now + patience_left, "renege", rid)
        self._serve(st)

    def _on_expire(self, rid):
        self.board.expire(self.now, rid)

    def _on_renege(self, rid):
        rec = self.recs[rid]
        if rec.state == "queue" and self.stations[rec.sid].leave(rid):
            rec.state, rec.outcome, rec.t_leave = "done", "abandoned", self.now
            self.closed.append(rid)

    def _on_finish(self, data):
        sid, idx, rid = data
        self.stations[sid].finish(idx, self.now)
        rec = self.recs[rid]
        rec.t_end, rec.state, rec.outcome = self.now, "done", "complete"
        self._serve(self.stations[sid])

    def _on_outage_start(self, o):
        self.stations[o.sid].outage_start(o.n_down, o.t_end)

    def _on_outage_end(self, o):
        self.stations[o.sid].outage_end(o.t_end)
        self._serve(self.stations[o.sid])

    def _serve(self, st: Station):
        for w, idx, t_fin in st.start_ready(self.now):
            rec = self.recs[w.rid]
            rec.t_start, rec.state = self.now, "charging"
            self.closed.append(w.rid)
            self._push(t_fin, "finish", (st.spec.sid, idx, w.rid))


def run(scenario, policy: Policy, cfg: SimConfig = DEFAULT_SIM) -> Simulation:
    return Simulation(scenario, policy, cfg).run()
