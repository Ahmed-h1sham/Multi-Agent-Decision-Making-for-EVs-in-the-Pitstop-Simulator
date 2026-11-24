"""Validation tests for the shared-charger extension (reuse map, "Build new")."""
import math
from dataclasses import replace

import pytest

from pitstop_mas.config import DEFAULT_DRIVERS, DEFAULT_SIM, DEMAND_LEVELS, StationSpec
from pitstop_mas.energy import charge_minutes, drive_energy_kwh
from pitstop_mas.metrics import run_metrics
from pitstop_mas.policies import (BOARD_INFO, INDEPENDENT, INDEPENDENT_DEFER, POLICIES,
                                  RESERVATION, RESERVATION_DEFER)
from pitstop_mas.scenario import Outage, Request, Scenario, generate, toy_herding
from pitstop_mas.sim import run
from pitstop_mas.stats import erlang_c_wq
from pitstop_mas.validation import simulate_mmc

LEVELS = ("normal", "high", "reduced")


def _scenario(stations, requests, outages=()):
    return Scenario("custom", 0, 0, 0.0, 0.0, tuple(stations), tuple(requests), tuple(outages))


def _toy_herding():
    return toy_herding()


# ---- queue core ---------------------------------------------------------------
def test_erlang_c_reference_values():
    # The values plotted in the background chapter (c = 2) and the "13x" figure.
    assert erlang_c_wq(2, 0.5) == pytest.approx(1 / 3, rel=1e-9)
    assert erlang_c_wq(2, 0.9) == pytest.approx(4.263, abs=1e-3)
    assert erlang_c_wq(2, 0.9) / erlang_c_wq(2, 0.5) == pytest.approx(12.79, abs=0.01)


def test_single_station_matches_erlang_c():
    v = simulate_mmc(2, 0.7, n=80_000, seed=11)
    assert v["ci_low"] <= v["analytic_wq"] <= v["ci_high"] or \
        abs(v["sim_wq"] - v["analytic_wq"]) / v["analytic_wq"] < 0.05
    assert v["max_busy"] <= 2 and v["violations"] == 0


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("pkey", list(POLICIES))
def test_capacity_never_exceeded(level, pkey):
    for j in range(4):
        sim = run(generate(level, j), POLICIES[pkey])
        for st in sim.stations:
            assert st.violations == []
            assert st.max_busy <= st.spec.n_chargers


@pytest.mark.parametrize("policy", [INDEPENDENT, BOARD_INFO, INDEPENDENT_DEFER])
def test_fcfs_order_kept(policy):
    for level in LEVELS:
        for j in range(4):
            sim = run(generate(level, j), policy)
            for st in sim.stations:
                keys = [k for _t, _rid, k in st.starts]
                assert keys == sorted(keys), f"FCFS broken at {st.spec.name}"


# ---- energy -------------------------------------------------------------------------
def test_faster_charger_same_energy_less_time():
    # Regression for reuse-map issue 1 (faster chargers delivered less energy).
    m7, m50, m150 = (charge_minutes(75.0, 0.20, 0.80, p) for p in (7.0, 50.0, 150.0))
    assert m150 < m50 < m7
    assert m50 == pytest.approx(0.6 * 75.0 / 50.0 * 60.0)
    full = (0.1 * 75 / (50 * 0.95) + 0.6 * 75 / 50 + 0.1 * 75 / (50 * 0.6)) * 60
    assert charge_minutes(75.0, 0.10, 0.90, 50.0) == pytest.approx(full)


def test_energy_balance():
    for level in LEVELS:
        sc = generate(level, 1)
        req = {r.rid: r for r in sc.requests}
        for p in POLICIES.values():
            sim = run(sc, p)
            for rec in sim.recs.values():
                if rec.sid is None:
                    continue
                r = req[rec.rid]
                expected_soc = r.soc0 - drive_energy_kwh(rec.road_km, r.soc0) / r.capacity_kwh
                assert rec.soc_arr == pytest.approx(expected_soc)
                assert rec.soc_arr >= DEFAULT_SIM.reserve_soc
                assert rec.energy_kwh == pytest.approx((r.soc_target - rec.soc_arr) * r.capacity_kwh)
                if rec.outcome == "complete":
                    assert rec.t_end - rec.t_start == pytest.approx(rec.dur)


# ---- coordination ---------------------------------------------------------------------
def test_toy_herding_matches_background_figure():
    sc = _toy_herding()
    a = run(sc, INDEPENDENT)
    b = run(sc, RESERVATION)
    assert [a.recs[i].sid for i in range(3)] == [0, 0, 0]      # all pick station A, B stays idle
    assert [b.recs[i].sid for i in range(3)] == [0, 1, 0]      # board: A - EV1, EV3; B - EV2
    assert run_metrics(b)["mean_w"] < run_metrics(a)["mean_w"]


@pytest.mark.parametrize("policy", [RESERVATION, RESERVATION_DEFER])
def test_booking_contract(policy):
    for level in LEVELS:
        for j in range(4):
            sim = run(generate(level, j), policy)
            recs = [r for r in sim.recs.values() if r.bseq is not None]
            # booking order follows the order in which vehicles commit
            recs.sort(key=lambda r: r.bseq)
            assert [r.t_decide for r in recs] == sorted(r.t_decide for r in recs)
            for r in recs:
                if r.t_arr is not None:
                    on_time = r.t_arr <= r.t_arr_est + DEFAULT_SIM.booking_tolerance + 1e-9
                    assert r.entitled == on_time
            # nothing left on the board except vehicles still driving
            for rid in sim.board.by_rid:
                assert sim.recs[rid].state == "travel"


@pytest.mark.parametrize("policy", [RESERVATION, RESERVATION_DEFER])
def test_entitled_vehicles_served_before_lapsed_ones(policy):
    for level in LEVELS:
        for j in range(6):
            sim = run(generate(level, j), policy)
            recs = list(sim.recs.values())
            for r in recs:
                if r.t_start is None or r.entitled:
                    continue
                for q in recs:
                    if q.sid != r.sid or not q.entitled or q.t_arr is None:
                        continue
                    waiting_then = q.t_arr < r.t_start - 1e-9 and (
                        (q.t_start is not None and q.t_start > r.t_start + 1e-9) or
                        (q.t_leave is not None and q.t_leave > r.t_start + 1e-9))
                    assert not waiting_then


def test_outage_reduces_capacity():
    st = StationSpec(0, "S", 5.0, 5.0, "dc_fast", 2)
    reqs = [Request(i, float(i), 5.0, 5.2, "Nissan Leaf", 62.0, 0.2, 0.8, 500.0, 1.0) for i in range(4)]
    sim = run(_scenario([st], reqs, [Outage(0, 1, 0.0, 300.0)]), INDEPENDENT)
    starts = sorted(r.t_start for r in sim.recs.values())
    # only one charger works during the outage, so starts are one session apart
    assert starts[1] - starts[0] == pytest.approx(sim.recs[0].dur, rel=0.05)


# ---- defer action ------------------------------------------------------------------------
def _one_station_two_evs():
    """EV 0 charges first; EV 1 requests 1 min later at the same spot and would queue ~43 min."""
    st = StationSpec(0, "S", 5.0, 5.0, "dc_fast", 1)
    reqs = [Request(i, float(i), 5.0, 5.2, "Nissan Leaf", 62.0, 0.2, 0.8, 500.0, 1.0) for i in range(2)]
    return _scenario([st], reqs)


def test_defer_is_just_in_time_departure():
    sc = _one_station_two_evs()
    a, ad = run(sc, INDEPENDENT), run(sc, INDEPENDENT_DEFER)
    r, rd = a.recs[1], ad.recs[1]
    step, k = DEFAULT_SIM.defer_step, DEFAULT_SIM.max_defer
    assert ad.recs[0].n_defer == 0                       # EV 0 finds a free charger
    assert rd.n_defer == k                               # EV 1 would queue anyway, so it waits at its spot
    assert rd.t_decide == pytest.approx(r.t_need + k * step)
    assert rd.d == pytest.approx(r.d)                    # same start of charging ...
    assert rd.w == pytest.approx(r.w - k * step)         # ... so only the place of waiting moved


def test_deferral_counts_from_the_original_request():
    step = DEFAULT_SIM.defer_step
    for policy in (INDEPENDENT_DEFER, RESERVATION_DEFER):
        for level in LEVELS:
            sim = run(generate(level, 1), policy)
            for rec in sim.recs.values():
                assert 0 <= rec.n_defer <= DEFAULT_SIM.max_defer
                assert rec.deferred == pytest.approx(rec.n_defer * step)
                if rec.t_decide is not None:
                    assert rec.t_decide == pytest.approx(rec.t_need + rec.n_defer * step)
                if rec.t_start is not None:
                    assert rec.d == pytest.approx(rec.t_start - rec.t_need)
                    assert rec.d >= rec.deferred


def test_deferring_vehicle_books_only_when_it_commits():
    for level in LEVELS:
        sim = run(generate(level, 2), RESERVATION_DEFER)
        for t, event, rid, _sid, _seq in sim.board.log:
            if event == "book":
                assert t == pytest.approx(sim.recs[rid].t_decide)


def test_deferral_uses_up_patience():
    for level in LEVELS:
        for j in range(4):
            sim = run(generate(level, j), RESERVATION_DEFER)
            req = {r.rid: r for r in sim.sc.requests}
            for rec in sim.recs.values():
                if rec.outcome == "abandoned":
                    assert rec.t_leave - rec.t_arr == pytest.approx(max(0.0, req[rec.rid].patience - rec.deferred))


def test_no_deferral_allowed_gives_the_base_policy():
    cfg = replace(DEFAULT_SIM, max_defer=0)
    for base, with_defer in ((INDEPENDENT, INDEPENDENT_DEFER), (RESERVATION, RESERVATION_DEFER)):
        for level in LEVELS:
            sc = generate(level, 0)
            m1, m2 = run_metrics(run(sc, base, cfg)), run_metrics(run(sc, with_defer, cfg))
            for m in (m1, m2):
                m.pop("decision_us"), m.pop("policy")
            assert m1 == m2


# ---- reproducibility and reporting ------------------------------------------------------
def test_scenarios_reproducible_and_policy_independent():
    for level in LEVELS:
        a, b = generate(level, 3), generate(level, 3)
        assert a.digest() == b.digest()
        assert generate(level, 4).digest() != a.digest()
        before = a.digest()
        for p in POLICIES.values():
            run(a, p)
        assert a.digest() == before


def test_runs_are_repeatable():
    sc = generate("high", 2)
    for p in POLICIES.values():
        m1, m2 = run_metrics(run(sc, p)), run_metrics(run(sc, p))
        m1.pop("decision_us"), m2.pop("decision_us")
        assert m1 == m2


def test_only_requests_in_the_measurement_window_are_counted():
    sc = generate("normal", 0)
    assert sc.t_warm == DEFAULT_SIM.t_warm and sc.t_meas_end == DEFAULT_SIM.t_arrivals_end
    assert all(r.t_need < DEFAULT_SIM.t_arrivals_end for r in sc.requests)
    assert any(r.t_need < sc.t_warm for r in sc.requests)          # warm-up requests exist ...
    m = run_metrics(run(sc, INDEPENDENT))
    assert m["n_req"] == sum(sc.measured(r.t_need) for r in sc.requests)   # ... but are not counted
    assert 0.0 < m["utilisation"] <= 1.0


def test_reduced_capacity_outage_lasts_the_whole_run():
    sc = generate("reduced", 0)
    assert sum(o.n_down for o in sc.outages) == DEMAND_LEVELS["reduced"].outage_chargers
    assert all(o.t_start == 0.0 and o.t_end > DEFAULT_SIM.t_end for o in sc.outages)
    sim = run(sc, INDEPENDENT)
    out = sum(not c.in_service or c.pending_out for st in sim.stations for c in st.chargers)
    assert out == DEMAND_LEVELS["reduced"].outage_chargers


def test_patience_within_the_fixed_range():
    lo, hi = DEFAULT_DRIVERS.patience_range
    for level in LEVELS:
        assert all(lo <= r.patience <= hi for r in generate(level, 1).requests)


def test_empty_denominator_is_undefined_not_zero():
    st = StationSpec(0, "S", 5.0, 5.0, "dc_fast", 1)
    m = run_metrics(run(_scenario([st], []), INDEPENDENT))
    assert m["n_req"] == 0
    assert math.isnan(m["completion"]) and math.isnan(m["mean_w"]) and math.isnan(m["p_tau"])
