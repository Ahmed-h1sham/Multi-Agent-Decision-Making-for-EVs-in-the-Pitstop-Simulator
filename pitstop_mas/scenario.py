"""Pre-generated scenarios (the external inputs both policies face).

Background, "Matched Scenarios and Reproducibility": a shared seed alone does not
guarantee identical inputs, because policy-dependent code can consume random numbers
in a different order. So every random draw happens here, once, from a generator that
belongs to the scenario (SeedSequence([base_seed, level, index])). The simulator and
the policies use no randomness at all. This also fixes issue 2 in the reuse map
(reset(seed=...) in the old environment did not make runs repeatable).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from .config import DEFAULT_DRIVERS, DEFAULT_SIM, DEMAND_LEVELS, EVAL_SEED, STATIONS, DriverModel, SimConfig
from .energy import CHARGER_POWER_KW, VEHICLE_BATTERY_KWH, charge_minutes

LEVEL_CODE = {"normal": 1, "high": 2, "reduced": 3, "custom": 9}
VEHICLES = tuple(VEHICLE_BATTERY_KWH)


@dataclass(frozen=True)
class Request:
    rid: int
    t_need: float        # charging need activated (minutes)
    x: float
    y: float
    vehicle: str
    capacity_kwh: float
    soc0: float          # SOC when the need is activated
    soc_target: float    # requested end SOC; belongs to the scenario, not the policy
    patience: float      # minutes the driver will queue before abandoning
    travel_noise: float  # actual travel time = estimated travel time * travel_noise


@dataclass(frozen=True)
class Outage:
    sid: int
    n_down: int
    t_start: float
    t_end: float


@dataclass(frozen=True)
class Scenario:
    level: str
    index: int
    seed: int
    rho: float
    arrival_rate_per_hour: float
    stations: tuple
    requests: tuple
    outages: tuple
    t_warm: float = 0.0              # requests activated before this are not measured
    t_meas_end: float = math.inf     # requests activated from this time on are not measured

    def measured(self, t_need: float) -> bool:
        return self.t_warm <= t_need < self.t_meas_end

    def digest(self) -> str:
        payload = json.dumps(
            {"r": [asdict(r) for r in self.requests], "o": [asdict(o) for o in self.outages],
             "s": [asdict(s) for s in self.stations], "w": [self.t_warm, self.t_meas_end]},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


_RATE_CACHE = {}


def nominal_service_rate_per_min(stations=STATIONS, drivers: DriverModel = DEFAULT_DRIVERS) -> float:
    """Sum over chargers of 1 / E[session minutes], estimated once from a fixed sample.

    Travel energy is ignored here, so this is a nominal figure used only to set the
    arrival rate for a target utilisation.
    """
    key = (tuple(stations), drivers)
    if key in _RATE_CACHE:
        return _RATE_CACHE[key]
    rng = np.random.default_rng(12345)
    n = 4000
    soc0 = rng.uniform(*drivers.soc0_range, n)
    caps = [VEHICLE_BATTERY_KWH[v] for v in rng.choice(VEHICLES, n)]
    rate = 0.0
    for st in stations:
        p = CHARGER_POWER_KW[st.charger_type]
        mean_min = float(np.mean([charge_minutes(c, s, drivers.soc_target, p) for c, s in zip(caps, soc0)]))
        rate += st.n_chargers / mean_min
    _RATE_CACHE[key] = rate
    return rate


def toy_herding() -> Scenario:
    """The background figure "Simultaneous choices": three EVs request charging 0.1 min
    apart from the same spot; station A is 2.6 min away, station B 7.8 min; one charger each."""
    from .config import StationSpec
    stations = (StationSpec(0, "A", 5.0, 5.0, "dc_fast", 1), StationSpec(1, "B", 7.0, 5.0, "dc_fast", 1))
    reqs = tuple(Request(i, 0.1 * i, 4.0, 5.0, "Tesla Model 3", 75.0, 0.20, 0.80, 120.0, 1.0) for i in range(3))
    return Scenario("custom", 0, 0, 0.0, 0.0, stations, reqs, ())


def generate(level: str, index: int, base_seed: int = EVAL_SEED, rho: float | None = None,
             sim: SimConfig = DEFAULT_SIM, drivers: DriverModel = DEFAULT_DRIVERS,
             stations: tuple = STATIONS) -> Scenario:
    spec = DEMAND_LEVELS.get(level)
    rho = rho if rho is not None else spec.rho
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, LEVEL_CODE.get(level, 9), index]))

    lam = rho * nominal_service_rate_per_min(stations, drivers)   # requests per minute

    requests = []
    lo, hi = 0.2, sim.area_km - 0.2
    t = rng.exponential(1.0 / lam)
    while t < sim.t_arrivals_end:
        if rng.random() < drivers.hotspot_share:
            x, y = rng.normal(drivers.hotspot, drivers.hotspot_sd_km)
        else:
            x, y = rng.uniform(lo, hi, 2)
        vehicle = str(rng.choice(VEHICLES))
        requests.append(Request(
            rid=len(requests),
            t_need=round(float(t), 3),
            x=round(float(np.clip(x, lo, hi)), 3),
            y=round(float(np.clip(y, lo, hi)), 3),
            vehicle=vehicle,
            capacity_kwh=VEHICLE_BATTERY_KWH[vehicle],
            soc0=round(float(rng.uniform(*drivers.soc0_range)), 4),
            soc_target=drivers.soc_target,
            patience=round(float(rng.uniform(*drivers.patience_range)), 2),
            travel_noise=round(float(rng.lognormal(0.0, drivers.travel_noise_sd)), 4),
        ))
        t += rng.exponential(1.0 / lam)

    outages = []
    if spec is not None and spec.outage_chargers > 0:
        slots = [st.sid for st in stations for _ in range(st.n_chargers)]
        picked = rng.choice(len(slots), size=spec.outage_chargers, replace=False)
        if spec.outage_minutes is None:          # out of service for the whole run
            start, end = 0.0, sim.t_end + 1.0
        else:                                    # a failure during the measurement window
            start = round(float(rng.uniform(sim.t_warm, sim.t_warm + 60.0)), 2)
            end = start + spec.outage_minutes
        per_station = {}
        for k in picked:
            per_station[slots[k]] = per_station.get(slots[k], 0) + 1
        for sid in sorted(per_station):
            outages.append(Outage(sid, per_station[sid], start, end))

    return Scenario(level=level, index=index, seed=base_seed, rho=rho,
                    arrival_rate_per_hour=round(lam * 60.0, 3), stations=tuple(stations),
                    requests=tuple(requests), outages=tuple(outages),
                    t_warm=sim.t_warm, t_meas_end=sim.t_arrivals_end)
