"""evaluate() must be pure, deterministic and self-describing."""

import pytest

from psrt.evaluate import compare, evaluate, torque_curve
from psrt.units import rpm_to_rad_s


def test_evaluate_does_not_mutate_the_state(state):
    before = state.fingerprint()
    log_length = len(state.log)
    evaluate(state, use_cache=False)
    assert state.fingerprint() == before
    assert len(state.log) == log_length


def test_evaluate_is_deterministic(state):
    a = evaluate(state, use_cache=False).to_dict()
    b = evaluate(state, use_cache=False).to_dict()
    assert a == b


def test_cached_and_uncached_agree(state):
    cached = evaluate(state, use_cache=True).to_dict()
    fresh = evaluate(state, use_cache=False).to_dict()
    assert cached == fresh


def test_metrics_are_json_serialisable(state):
    import json
    json.dumps(evaluate(state).to_dict())


def test_energy_closure_is_reported_and_passes(state):
    m = evaluate(state)
    assert m.checks["energy_closure_pass"]
    assert m.checks["energy_closure_error"] < 1e-3


def test_invalid_state_is_refused_rather_than_silently_evaluated(state):
    state.param("masses.piston").value = None
    with pytest.raises(ValueError, match="not valid"):
        evaluate(state, use_cache=False)


def test_bore_headroom_is_always_reported(state):
    notes = " ".join(evaluate(state).notes)
    assert "headroom" in notes


def test_model_limitations_are_declared_not_hidden(state):
    notes = " ".join(evaluate(state).notes)
    assert "closed-form" in notes and "phase 6" in notes


def test_structural_summary_is_reported(state):
    st = evaluate(state).structural
    assert st["margin_count"] >= 20
    assert st["binding_component"]
    assert st["binding_operating_component"]
    assert set(st["by_component"]) >= {"crown", "rod", "pin", "sleeve"}


def test_short_rod_raises_a_warning(state):
    state.set("engine.rod_length", state["engine.crank_radius"] * 2.6)
    assert any("rod ratio" in w for w in evaluate(state).warnings)


def test_overspeed_raises_a_piston_speed_warning(state):
    state.set("operating.speed", rpm_to_rad_s(11000))
    assert any("piston speed" in w for w in evaluate(state).warnings)


def test_torque_curve_peaks_near_the_volumetric_efficiency_peak(state):
    ve_rpm = state["operating.ve_peak_speed"] * 60.0 / (2.0 * 3.141592653589793)
    tc = torque_curve(state, range(1500, 7501, 250))
    assert abs(tc["peak_torque_rpm"] - ve_rpm) < 750.0
    assert tc["peak_power_rpm"] > tc["peak_torque_rpm"]


def test_torque_curve_leaves_the_state_untouched(state):
    before = state.fingerprint()
    torque_curve(state, [2000, 4000, 6000])
    assert state.fingerprint() == before


def test_compare_reports_both_improvement_and_cost(state):
    base = evaluate(state)
    bigger = evaluate(state.with_changes({"engine.bore": 0.0880}, "ai", "overbore"))
    d = compare(base, bigger)

    assert d["performance.indicated_torque_nm"]["delta"] > 0
    assert d["loads.peak_side_thrust_n"]["delta"] > 0
    assert d["geometry.displacement_total_m3"]["percent"] > 4.0


def test_compare_of_a_state_with_itself_is_empty(state):
    assert compare(evaluate(state), evaluate(state)) == {}
