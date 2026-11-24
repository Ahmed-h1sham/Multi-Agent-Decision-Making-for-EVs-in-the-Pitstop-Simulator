"""Station-choice policies and the shared reservation board.

All three policies use the same public station information and the same objective:
choose the reachable station with the earliest projected start of charging
(travel + queue wait = the charging-access delay d_i). Ties go to the shorter trip,
then the lower station id. The only differences are:

  A       Independent choice: no board; stations serve FCFS.
  B_info  Board visible (vehicles see who is already on the way), stations serve FCFS.
          Isolates the information effect of the board.
  B       Reservation board: board visible and valid bookings are served in booking
          order. Information effect + priority effect.

A vs B is the core comparison of the thesis. B_info is the secondary comparison
("a further policy that uses the board with the baseline service rule").

A_d and B_d add the defer action to A and B: an EV may wait where it is and decide
again later with newer information (see Policy.decide).
"""
from __future__ import annotations

from dataclasses import dataclass

from .station import BOOKING, FCFS, INF, project_start


@dataclass
class Booking:
    rid: int
    sid: int
    bseq: int          # board sequence number = booking order
    t_booked: float
    t_arr_est: float
    dur: float
    deadline: float    # t_arr_est + tolerance; after this the booking lapses


class ReservationBoard:
    """Shared reservation board (an indirect communication channel).

    Service contract used in this demo:
      resource booked     a place at a station (vehicles choose stations, not chargers)
      visibility          a booking is visible to every later decision immediately
      validity            from the booking until the vehicle arrives
      arrival tolerance   `tolerance` minutes after the expected arrival time
      expiry / no-show    an unused booking lapses at its deadline; the vehicle then
                          counts as a walk-in when it arrives
      cancellation        a driver who abandons the queue frees the place
      conflicts           decisions are processed one at a time in event order, so two
                          vehicles never book "at the same moment"; ties in time are
                          broken by request id
      service entitlement set by the station rule: 'booking' serves valid bookings in
                          booking order (Policy B); 'fcfs' ignores them (Policy B_info)
    """

    def __init__(self, tolerance: float):
        self.tolerance = tolerance
        self.enroute: dict[int, dict[int, Booking]] = {}
        self.by_rid: dict[int, Booking] = {}
        self.next_seq = 0
        self.log: list[tuple] = []   # (t, event, rid, sid, bseq)

    def book(self, t: float, rid: int, sid: int, t_arr_est: float, dur: float) -> Booking:
        b = Booking(rid, sid, self.next_seq, t, t_arr_est, dur, t_arr_est + self.tolerance)
        self.next_seq += 1
        self.enroute.setdefault(sid, {})[rid] = b
        self.by_rid[rid] = b
        self.log.append((t, "book", rid, sid, b.bseq))
        return b

    def arrived(self, t: float, rid: int) -> Booking | None:
        """Vehicle reached its station; returns the booking if it is still valid."""
        b = self.by_rid.pop(rid, None)
        if b is None:
            return None
        del self.enroute[b.sid][rid]
        self.log.append((t, "arrive", rid, b.sid, b.bseq))
        return b

    def expire(self, t: float, rid: int) -> bool:
        b = self.by_rid.pop(rid, None)
        if b is None:
            return False
        del self.enroute[b.sid][rid]
        self.log.append((t, "lapse", rid, b.sid, b.bseq))
        return True

    def arriving(self, sid: int, rule: str) -> list:
        """En-route bookings at a station, as (t_arr_est, service key, duration)."""
        out = []
        for b in self.enroute.get(sid, {}).values():
            key = (0, b.bseq, 0) if rule == BOOKING else (1, b.t_arr_est, 1e12 + b.bseq)
            out.append((b.t_arr_est, key, b.dur))
        return out


@dataclass(frozen=True)
class Option:
    sid: int
    road_km: float
    travel_est: float   # minutes
    soc_arr: float
    dur: float          # session minutes at this station


GO = "go"
DEFER = "defer"


@dataclass(frozen=True)
class Decision:
    action: str          # GO or DEFER
    sid: int | None      # chosen station (GO only)
    est: float | None    # projected start at the chosen station (GO only)
    seen: dict           # projected start per reachable station at decision time


@dataclass(frozen=True)
class Policy:
    """What an EV may see and do, and the rule it decides with.

    uses_board    the EV sees the reservation board (en-route bookings)
    service_rule  how stations serve: FCFS, or valid bookings first (BOOKING)
    can_defer     the EV may wait where it is and decide again later
    """
    key: str
    label: str
    uses_board: bool
    service_rule: str
    can_defer: bool = False

    def my_key(self, sim, me_arr: float) -> tuple:
        if self.uses_board and self.service_rule == BOOKING:
            return (0, sim.board.next_seq, 0)
        return (1, me_arr, INF)

    def estimates(self, sim, options: list) -> dict:
        """Projected start of charging at each reachable station, from this policy's information."""
        seen = {}
        for opt in options:
            st = sim.stations[opt.sid]
            arriving = sim.board.arriving(opt.sid, self.service_rule) if self.uses_board else []
            me_arr = sim.now + opt.travel_est
            seen[opt.sid] = project_start(sim.now, st.public_free_times(sim.now), st.public_queue(),
                                          arriving, me_arr, self.my_key(sim, me_arr), opt.dur)
        return seen

    def choose(self, sim, options: list) -> tuple:
        """Return (station id, projected start, {sid: projected start}) for the deciding vehicle."""
        seen = self.estimates(sim, options)
        best = min(options, key=lambda o: (seen[o.sid], o.travel_est, o.sid))
        return best.sid, seen[best.sid], seen

    def decide(self, sim, rec, options: list) -> Decision:
        """Choose a station now, or defer.

        Defer rule ("just-in-time departure"): if the EV would wait at least defer_step
        minutes in the queue anyway, it waits where it is instead and decides again after
        defer_step minutes, keeping all stations open. At most max_defer times. The cost is
        that vehicles deciding in between can take its place; the delay d keeps counting
        from the original request.
        """
        sid, est, seen = self.choose(sim, options)
        if self.can_defer and rec.n_defer < sim.cfg.max_defer:
            travel = next(o.travel_est for o in options if o.sid == sid)
            if est - (sim.now + travel) >= sim.cfg.defer_step:
                return Decision(DEFER, None, None, seen)
        return Decision(GO, sid, est, seen)


INDEPENDENT = Policy("A", "Independent choice", uses_board=False, service_rule=FCFS)
BOARD_INFO = Policy("B_info", "Board info, FCFS service", uses_board=True, service_rule=FCFS)
RESERVATION = Policy("B", "Reservation board", uses_board=True, service_rule=BOOKING)
INDEPENDENT_DEFER = Policy("A_d", "Independent choice + defer", uses_board=False,
                           service_rule=FCFS, can_defer=True)
RESERVATION_DEFER = Policy("B_d", "Reservation board + defer", uses_board=True,
                           service_rule=BOOKING, can_defer=True)

POLICIES = {p.key: p for p in (INDEPENDENT, BOARD_INFO, RESERVATION, INDEPENDENT_DEFER, RESERVATION_DEFER)}
