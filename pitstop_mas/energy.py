"""Energy and charging models.

Reused from the earlier Pitstop EV adaptation (Osman, 2026), file env/gym_ev_env.py:
  * base consumption of 0.18 kWh/km and the SOC factor 1 + (1 - SOC) * 0.05
  * the charger power table (Level 2 7 kW, DC fast 50 kW, ultra-fast 150 kW)
  * the SOC-band charging factors 0.95 (below 20 %), 1.0 (20-80 %), 0.6 (above 80 %)
  * the vehicle battery sizes listed in streamlit_app_ev.py

Corrected (issues 1, 4 and 5 in claude/ev_simulator_reuse_map.md):
  * There is no time_multiplier. Session time is the integral of dE / P(SOC), so a
    faster charger always delivers the same energy in less time. In the old code the
    multiplier was applied on top of the power, so a DC fast charger stopped at 44 %.
  * The band factor applies to the SOC the battery is in at each moment, not only to
    the SOC at which charging started.
  * Driving to a station uses energy.
  * Each vehicle keeps its own battery size; the old environment used 75 kWh for all.
"""

BASE_KWH_PER_KM = 0.18

CHARGER_POWER_KW = {
    "level2": 7.0,
    "dc_fast": 50.0,
    "ultra_fast": 150.0,
}

VEHICLE_BATTERY_KWH = {
    "Tesla Model 3": 75.0,
    "Nissan Leaf": 62.0,
    "BMW i4": 83.0,
    "Hyundai Ioniq": 77.0,
    "Ford Mustang Mach-E": 88.0,
}

_BANDS = ((0.0, 0.20, 0.95), (0.20, 0.80, 1.0), (0.80, 1.0, 0.6))


def band_factor(soc: float) -> float:
    """Share of the charger's rated power that the battery accepts at this SOC."""
    for lo, hi, factor in _BANDS:
        if lo <= soc < hi:
            return factor
    return _BANDS[-1][2]


def drive_energy_kwh(road_km: float, soc: float) -> float:
    """Energy to drive road_km, starting at state of charge soc (0-1)."""
    return road_km * BASE_KWH_PER_KM * (1.0 + (1.0 - soc) * 0.05)


def charge_minutes(capacity_kwh: float, soc_from: float, soc_to: float, power_kw: float) -> float:
    """Minutes to charge from soc_from to soc_to at a charger rated power_kw.

    Exact integral over the piecewise-constant SOC bands.
    """
    if soc_to <= soc_from:
        return 0.0
    hours = 0.0
    for lo, hi, factor in _BANDS:
        a, b = max(lo, soc_from), min(hi, soc_to)
        if b > a:
            hours += (b - a) * capacity_kwh / (power_kw * factor)
    return hours * 60.0


def charge_energy_kwh(capacity_kwh: float, soc_from: float, soc_to: float) -> float:
    return max(0.0, soc_to - soc_from) * capacity_kwh
