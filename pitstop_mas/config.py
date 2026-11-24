"""Configuration values.

The method values (warm-up, measurement window, tau, patience, demand levels) were fixed
on 3 Nov 2026 before the final evaluation run; the reasons and sources are in README.md,
section "Method values". Times are in minutes from the start of a scenario; distances in km.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StationSpec:
    sid: int
    name: str
    x: float
    y: float
    charger_type: str
    n_chargers: int


# Urban area of 12 km x 12 km with five stations (10 chargers in total).
STATIONS = (
    StationSpec(0, "Downtown", 6.0, 6.0, "dc_fast", 3),
    StationSpec(1, "North", 5.0, 10.0, "dc_fast", 2),
    StationSpec(2, "East", 10.0, 7.0, "ultra_fast", 1),
    StationSpec(3, "South", 7.0, 2.0, "dc_fast", 2),
    StationSpec(4, "West", 2.0, 5.0, "dc_fast", 2),
)


@dataclass(frozen=True)
class SimConfig:
    area_km: float = 12.0
    detour_factor: float = 1.3          # road km per straight-line km
    minutes_per_road_km: float = 2.0    # 30 km/h urban average
    reserve_soc: float = 0.05           # a station is reachable only if this SOC is left on arrival
    t_warm: float = 240.0               # warm-up: requests before this are simulated, not measured
    t_obs: float = 360.0                # measurement window: requests in [t_warm, t_warm + t_obs)
    t_drain: float = 120.0              # run continues after the last request so sessions can finish
    tau: float = 15.0                   # excessive-waiting threshold (fixed before testing)
    booking_tolerance: float = 10.0     # minutes after the expected arrival before a booking lapses
    defer_step: float = 5.0             # minutes a deferring EV waits before deciding again
    max_defer: int = 3                  # deferrals allowed per request

    @property
    def t_arrivals_end(self) -> float:
        """No request is activated after this time."""
        return self.t_warm + self.t_obs

    @property
    def t_end(self) -> float:
        """The run stops here; requests still open are reported as unfinished."""
        return self.t_warm + self.t_obs + self.t_drain


@dataclass(frozen=True)
class DemandLevel:
    name: str
    rho: float                          # nominal utilisation lambda / (sum of charger service rates)
    outage_chargers: int = 0            # chargers out of service in the scenario
    outage_minutes: float | None = None # None: out for the whole run; otherwise length of the outage


DEMAND_LEVELS = {
    "normal": DemandLevel("normal", rho=0.60),
    "high": DemandLevel("high", rho=0.90),
    # same demand as normal, with 3 of the 10 chargers (30 %) out of service for the whole run
    "reduced": DemandLevel("reduced", rho=0.60, outage_chargers=3),
}


@dataclass(frozen=True)
class DriverModel:
    """Distributions the scenario generator draws from (scenario-owned, never policy-owned)."""
    hotspot: tuple = (6.0, 6.0)
    hotspot_share: float = 0.55
    hotspot_sd_km: float = 1.8
    soc0_range: tuple = (0.08, 0.35)
    soc_target: float = 0.80
    patience_range: tuple = (15.0, 60.0)   # minutes a driver will queue before leaving
    travel_noise_sd: float = 0.15          # log-normal factor on travel time


DEFAULT_SIM = SimConfig()
DEFAULT_DRIVERS = DriverModel()

# Scenario base seeds. Each set is used for one purpose only.
DEV_SEED = 2026     # development: used while building the model and choosing these values
WELCH_SEED = 777    # warm-up analysis (Welch's procedure)
TRAIN_SEED = 1000   # training scenarios for the learned policy
EVAL_SEED = 2027    # final evaluation, first used after the values above were fixed
