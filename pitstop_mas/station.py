"""Stations, chargers, the queue and the service rule.

A station has c chargers that serve one queue (background, Figure "Station, charger
and queue model"). A vehicle chooses a station, not a charger. The service rule decides
who is served when a charger becomes free:

  fcfs     first come, first served by physical arrival time (Policy A and B-info)
  booking  vehicles holding a valid booking are served in booking order; vehicles whose
           booking lapsed are served after them, by arrival time (Policy B)

Both rules are work-conserving: a charger is never held idle while a vehicle is present.
Holding a charger for a booked vehicle that has not arrived yet is the other contract
option named in the background; it is not implemented in this demo.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

FCFS = "fcfs"
BOOKING = "booking"
INF = math.inf


@dataclass
class Charger:
    idx: int
    rid: int | None = None      # vehicle charging now
    free_at: float = 0.0        # end of the current session
    in_service: bool = True
    pending_out: bool = False   # goes out of service when the current session ends
    back_at: float = INF        # announced return time while out of service


@dataclass
class Waiting:
    key: tuple                  # smaller key is served first
    rid: int
    t_arr: float
    dur: float                  # session minutes this vehicle needs here


@dataclass
class Station:
    spec: object
    rule: str
    chargers: list = field(default_factory=list)
    queue: list = field(default_factory=list)
    starts: list = field(default_factory=list)      # (t, rid, key) in start order
    violations: list = field(default_factory=list)  # capacity-rule breaches (must stay empty)
    max_busy: int = 0

    def __post_init__(self):
        if not self.chargers:
            self.chargers = [Charger(i) for i in range(self.spec.n_chargers)]

    # ---- state -------------------------------------------------------------
    @property
    def busy(self) -> int:
        return sum(c.rid is not None for c in self.chargers)

    def idle_available(self) -> int:
        return sum(c.in_service and not c.pending_out and c.rid is None for c in self.chargers)

    # ---- queue ---------------------------------------------------------------
    def join(self, w: Waiting) -> None:
        self.queue.append(w)

    def leave(self, rid: int) -> bool:
        for i, w in enumerate(self.queue):
            if w.rid == rid:
                del self.queue[i]
                return True
        return False

    def start_ready(self, t: float) -> list:
        """Start sessions while a charger is idle and someone is waiting."""
        started = []
        while self.queue:
            idle = [c for c in self.chargers if c.in_service and not c.pending_out and c.rid is None]
            if not idle:
                break
            w = min(self.queue, key=lambda q: q.key)
            self.queue.remove(w)
            c = idle[0]
            c.rid, c.free_at = w.rid, t + w.dur
            self.starts.append((t, w.rid, w.key))
            busy = self.busy
            self.max_busy = max(self.max_busy, busy)
            out_now = sum(not ch.in_service for ch in self.chargers)
            if busy > self.spec.n_chargers - out_now:
                self.violations.append((t, busy, out_now))
            started.append((w, c.idx, c.free_at))
        return started

    def finish(self, idx: int, t: float) -> None:
        c = self.chargers[idx]
        c.rid, c.free_at = None, t
        if c.pending_out:
            c.in_service, c.pending_out = False, False

    # ---- outages -------------------------------------------------------------
    def outage_start(self, n: int, t_back: float) -> None:
        """Take n chargers out of service: idle ones at once, busy ones after their session."""
        taken = 0
        for c in sorted(self.chargers, key=lambda ch: ch.rid is not None):
            if taken == n:
                break
            if not c.in_service or c.pending_out:
                continue
            if c.rid is None:
                c.in_service = False
            else:
                c.pending_out = True
            c.back_at = t_back
            taken += 1

    def outage_end(self, t_back: float) -> None:
        for c in self.chargers:
            if c.back_at == t_back:
                c.in_service, c.pending_out, c.back_at = True, False, INF

    # ---- public information ----------------------------------------------------
    def public_free_times(self, now: float) -> list:
        """When each charger can next take a vehicle, as a charging app would show it."""
        out = []
        for c in self.chargers:
            if not c.in_service or c.pending_out:
                out.append(max(c.back_at, c.free_at if c.rid is not None else now))
            elif c.rid is None:
                out.append(now)
            else:
                out.append(c.free_at)
        return out

    def public_queue(self) -> list:
        return [(w.key, w.dur) for w in self.queue]


def project_start(now: float, free_times: list, present: list, arriving: list,
                  me_arr: float, me_key: tuple, me_dur: float) -> float:
    """Projected start time for a deciding vehicle at one station.

    Replays the station's own service rule over the information given:
      free_times  when each charger can next take a vehicle
      present     (key, dur) of vehicles already queued
      arriving    (t_arr, key, dur) of vehicles known to be on their way (board only)
    Policy A passes arriving=[]; that missing list is the whole difference between the
    independent and the board-based estimate.
    """
    F = sorted(max(f, now) for f in free_times)
    if not F or F[0] == INF:
        return INF
    heapq.heapify(F)
    pool = [(k, 0, d) for k, d in present]
    heapq.heapify(pool)
    pend = sorted([(a, k, d, 0) for a, k, d in arriving] + [(me_arr, me_key, me_dur, 1)],
                  key=lambda p: (p[0], p[1], p[3]))
    i = 0
    while True:
        f = F[0]
        if f == INF:
            return INF
        while i < len(pend) and pend[i][0] <= f:
            heapq.heappush(pool, (pend[i][1], pend[i][3], pend[i][2]))
            i += 1
        if pool:
            t_serve = f
        else:
            a = pend[i][0]
            while i < len(pend) and pend[i][0] <= a:
                heapq.heappush(pool, (pend[i][1], pend[i][3], pend[i][2]))
                i += 1
            t_serve = a
        _key, is_me, dur = heapq.heappop(pool)
        if is_me:
            return t_serve
        heapq.heapreplace(F, t_serve + dur)
