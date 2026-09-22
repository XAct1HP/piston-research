"""The backend: every endpoint, and the semantics that matter.

The propose / commit / revert cycle is tested hardest, because phase 5 hands
it to an AI. The property that makes that safe is that a proposal evaluates
against a copy: the live design must be untouched until commit, and a refused
write must come back as a structured reason rather than a 500.
"""

import os

import pytest
from fastapi.testclient import TestClient

from psrt.server.app import create_app
from psrt.server.session import Session

HERE = os.path.dirname(os.path.abspath(__file__))
LS3 = os.path.join(os.path.dirname(HERE), "examples", "ls3.json")


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(LS3))


def get(client, path):
    r = client.get(path)
    assert r.status_code == 200, (path, r.status_code)
    return r.json()


# --- reading ---------------------------------------------------------------

def test_state_carries_bounds_locks_and_reasons(client):
    state = get(client, "/api/state")
    assert state["name"] == "GM LS3 6.2L V8"
    flat = {p["path"]: p for s in state["sections"] for p in s["parameters"]}

    bore = flat["engine.bore"]
    assert bore["class"] == "bounded"
    assert bore["editable"] is True
    assert "subtractive" in bore["why_min"]

    assert flat["block.as_built_bore"]["class"] == "locked"
    assert flat["block.as_built_bore"]["editable"] is False
    assert flat["engine.rod_ratio"]["class"] == "derived"
    assert flat["engine.rod_ratio"]["editable"] is False


def test_derived_parameters_carry_their_computed_value(client):
    flat = {p["path"]: p for s in get(client, "/api/state")["sections"]
            for p in s["parameters"]}
    assert flat["engine.rod_ratio"]["value"] == pytest.approx(3.349, abs=0.01)


def test_sweep_gives_the_viewport_real_kinematics(client):
    sweep = get(client, "/api/sweep")
    travel = max(sweep["piston_translate_mm"]) - min(sweep["piston_translate_mm"])
    assert travel == pytest.approx(92.0, abs=0.01), "piston travel is the stroke"
    assert len(sweep["rod_rotation_y"]) == len(sweep["theta_deg"])
    assert sweep["rpm"] == pytest.approx(4600, abs=1)


def test_rod_rotation_is_continuous_through_tdc(client):
    """It is written as pi - phi rather than atan2 on the rod vector, so it
    does not wrap at +/-180 and the viewport can interpolate between samples."""
    ry = get(client, "/api/sweep")["rod_rotation_y"]
    steps = [abs(b - a) for a, b in zip(ry, ry[1:])]
    assert max(steps) < 0.05, "no wrap-around jump"


def test_sweep_arrays_are_all_the_same_length(client):
    sweep = get(client, "/api/sweep")
    keys = ["theta_deg", "pressure_bar", "f_gas_kn", "f_pin_kn", "f_side_kn",
            "torque_nm", "piston_translate_mm", "pin_z_mm", "rod_rotation_y"]
    assert len({len(sweep[k]) for k in keys}) == 1


def test_margins_and_metrics(client):
    margins = get(client, "/api/margins")
    assert len(margins["margins"]) == 25
    assert margins["binding"]
    assert margins["binding_operating"]

    metrics = get(client, "/api/metrics")
    assert metrics["structural"]["binding_component"]
    assert metrics["checks"]["energy_closure_pass"] is True


def test_geometry_returns_meshes_and_properties(client):
    g = get(client, "/api/geometry?tolerance=1.0")
    assert g["ok"] is True
    assert set(g["meshes"]) == {"piston", "pin", "rod", "sleeve"}
    for part, mesh in g["meshes"].items():
        assert mesh["triangle_count"] > 0, part
    assert g["properties"]["piston"]["mass_kg"] > 0


def test_profiles_endpoint_answers_the_hexagon_question(client):
    study = get(client, "/api/profiles")
    rows = {r["key"]: r for r in study["rows"]}
    assert rows["circle"]["fraction_of_circle"] == pytest.approx(1.0)
    assert rows["polygon-6"]["fraction_of_circle"] == pytest.approx(0.827, abs=1e-3)


def test_curve_endpoint(client):
    curve = get(client, "/api/curve?start=3000&stop=5000&step=1000")
    assert curve["rpm"] == [3000.0, 4000.0, 5000.0]
    assert curve["peak_brake_torque_nm"] > 0


def test_curve_rejects_a_nonsense_range(client):
    assert client.get("/api/curve?start=5000&stop=1000").status_code == 400


def test_no_nan_or_infinity_reaches_the_browser(client):
    """JSON has no NaN. Anything non-finite has to arrive as null or the
    front end silently breaks on it."""
    import json
    for path in ("/api/metrics", "/api/margins", "/api/sweep"):
        raw = client.get(path).text
        assert "NaN" not in raw and "Infinity" not in raw, path
        json.loads(raw)


# --- writing ---------------------------------------------------------------

def test_a_locked_write_is_refused_with_its_reason(client):
    r = client.post("/api/state",
                    json={"changes": {"block.as_built_bore": 0.11}})
    assert r.status_code == 200, "a refusal is an answer, not a server error"
    body = r.json()
    assert body["ok"] is False
    assert body["violation"]["path"] == "block.as_built_bore"
    assert "the block exists" in body["violation"]["reason"]


def test_an_out_of_bounds_write_is_refused_with_its_bound(client):
    body = client.post("/api/state",
                       json={"changes": {"engine.bore": 0.20}}).json()
    assert body["ok"] is False
    envelope_max = client.get("/api/envelope").json()["max_bore_m"]
    assert body["violation"]["bound"] == pytest.approx(envelope_max, abs=1e-9)


def test_a_refused_write_leaves_the_design_untouched(client):
    before = get(client, "/api/state")["fingerprint"]
    client.post("/api/state", json={"changes": {"engine.bore": 0.20}})
    assert get(client, "/api/state")["fingerprint"] == before


def test_a_permitted_write_applies_and_can_be_undone(client):
    before = get(client, "/api/state")["fingerprint"]
    body = client.post("/api/state", json={
        "changes": {"piston.crown_thickness": 0.0072},
        "rationale": "test"}).json()
    assert body["ok"] is True
    assert get(client, "/api/state")["fingerprint"] != before

    assert client.post("/api/revert").json()["ok"] is True
    assert get(client, "/api/state")["fingerprint"] == before


# --- propose / commit / revert --------------------------------------------

def test_a_proposal_does_not_touch_the_live_design(client):
    """The property that makes it safe to let an AI drive in phase 5."""
    before = get(client, "/api/state")["fingerprint"]
    body = client.post("/api/propose", json={
        "changes": {"engine.bore": 0.1035},
        "rationale": "chasing displacement"}).json()
    assert body["ok"] is True
    assert get(client, "/api/state")["fingerprint"] == before
    client.post(f"/api/discard/{body['proposal_id']}")


def test_a_proposal_reports_what_got_worse(client):
    body = client.post("/api/propose", json={
        "changes": {"engine.bore": 0.1035}, "rationale": "overbore"}).json()
    names = [r["metric"] for r in body["regressions"]]
    assert any("side_thrust" in n for n in names)
    assert any("peak_pin" in n for n in names)
    assert body["deltas"]["performance.indicated_torque_nm"]["delta"] > 0
    client.post(f"/api/discard/{body['proposal_id']}")


def test_committing_a_proposal_applies_exactly_it(client):
    before = get(client, "/api/state")["fingerprint"]
    proposal = client.post("/api/propose", json={
        "changes": {"piston.crown_thickness": 0.0075},
        "rationale": "thicker crown"}).json()

    assert client.post(f"/api/commit/{proposal['proposal_id']}").json()["ok"]
    flat = {p["path"]: p for s in get(client, "/api/state")["sections"]
            for p in s["parameters"]}
    assert flat["piston.crown_thickness"]["value"] == pytest.approx(0.0075)

    client.post("/api/revert")
    assert get(client, "/api/state")["fingerprint"] == before


def test_a_refused_proposal_returns_the_violation(client):
    body = client.post("/api/propose", json={
        "changes": {"engine.bore": 0.20}, "rationale": "absurd"}).json()
    assert body["ok"] is False
    assert "violation" in body


def test_committing_an_unknown_proposal_fails_cleanly(client):
    assert client.post("/api/commit/nosuchproposal").json()["ok"] is False


def test_proposals_are_atomic_across_several_parameters(client):
    before = get(client, "/api/state")["fingerprint"]
    body = client.post("/api/propose", json={
        "changes": {"piston.crown_thickness": 0.0071,
                    "block.as_built_bore": 0.11},
        "rationale": "one of these is locked"}).json()
    assert body["ok"] is False
    assert get(client, "/api/state")["fingerprint"] == before


# --- the page --------------------------------------------------------------

def test_the_page_and_its_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "Piston System Research Tool" in page.text
    for asset in ("/static/app.js", "/static/styles.css",
                  "/static/vendor/three.min.js",
                  "/static/vendor/OrbitControls.js"):
        assert client.get(asset).status_code == 200, asset


def test_three_js_is_vendored_not_fetched_from_a_cdn(client):
    """An engineering tool should not need the internet to draw a piston."""
    assert "cdn" not in client.get("/").text.lower()


# --- the session object ----------------------------------------------------

def test_session_undo_stack_is_bounded_to_real_changes():
    session = Session.from_file(LS3)
    assert session.revert()["ok"] is False
    session.set({"piston.crown_thickness": 0.0068}, rationale="test")
    assert session.revert()["ok"] is True
    assert session.revert()["ok"] is False


def test_the_fea_endpoint_returns_a_drawable_field(client):
    body = client.get("/api/fea/pin?elements=8000").json()
    assert body["ok"], body
    field = body["field"]
    assert len(field["vertices"]) == len(field["stress_mpa"])
    assert field["range_mpa"]["clip"] <= field["range_mpa"]["max"]
    assert body["residual"] < 1e-9
    assert body["free_rigid_modes"] == []


def test_an_unknown_part_is_refused_with_what_is_available(client):
    body = client.get("/api/fea/crankshaft").json()
    assert body["ok"] is False
    assert "pin" in body["error"]


def test_the_case_list_is_not_swallowed_by_the_part_route(client):
    """FastAPI matches routes in declaration order, so /api/fea/cases has to
    be declared before /api/fea/{part} or it answers "no load case for
    'cases'". The front end builds its buttons from this."""
    body = client.get("/api/fea/cases").json()
    assert body["ok"]
    assert set(body["parts"]) == {"pin", "piston", "rod", "sleeve"}


def test_a_solver_failure_is_reported_not_a_500(client, monkeypatch):
    """A bare 500 reaches the browser as "Solve failed: -> 500" and sends
    someone hunting through a traceback. Failures come back as a message."""
    import psrt.fea.cases as cases_mod

    def explode(*a, **k):
        raise RuntimeError("contrived")

    monkeypatch.setitem(cases_mod.CASES, "pin", explode)
    body = client.get("/api/fea/pin").json()
    assert body["ok"] is False
    assert "contrived" in body["error"]


def test_the_envelope_endpoint_carries_its_confidence(client):
    body = client.get("/api/envelope").json()
    assert body["ok"]
    assert body["binding"]["name"]
    assert body["confidence_tier"] in (1, 2, 3)
    assert isinstance(body["unknown"], list)


def test_an_overbore_past_the_block_is_refused_with_a_reason(client):
    body = client.get("/api/overbore?bore_mm=110").json()
    assert body["ok"] is False
    assert body["refused"] is True
    assert "bore" in body["error"]


def test_an_allowed_overbore_reports_both_directions(client):
    body = client.get("/api/overbore?bore_mm=103.5&fast=true").json()
    assert body["ok"]
    assert body["better"] and body["worse"]


def test_the_optimiser_endpoint_does_not_commit(client):
    """A searched design is a proposal, not a change. The live design has to
    be untouched when the call returns."""
    before = get(client, "/api/state")["fingerprint"]
    body = client.get("/api/optimise?objective=torque&levers=4").json()
    assert body["ok"], body
    assert body["warnings"]
    assert get(client, "/api/state")["fingerprint"] == before


def test_an_unknown_objective_is_refused_not_500(client):
    body = client.get("/api/optimise?objective=vibes").json()
    assert body["ok"] is False
    assert "unknown objective" in body["error"]


def test_the_frontier_endpoint_returns_priced_points(client):
    body = client.get("/api/frontier?objective=torque&points=3&levers=3").json()
    assert body["ok"], body
    assert len(body["points"]) == 3
    assert any(p["feasible"] for p in body["points"])


def test_an_impossible_combination_is_refused_not_a_500(client):
    """A value can be inside its own bounds and still make the design
    impossible: a rod flange within range but too thick for the shank height
    it sits in. That only surfaces when something derived from it is
    computed, which used to happen after the write had landed -- so the
    browser got a 500 and the design was left holding a value it could not
    evaluate."""
    before = get(client, "/api/state")["fingerprint"]
    body = client.post("/api/state",
                       json={"changes": {"rod.shank_flange": 0.020}}).json()

    assert body["ok"] is False
    assert body["violation"]["error"] == "impossible_combination"
    assert "flanges" in body["violation"]["reason"]
    assert get(client, "/api/state")["fingerprint"] == before


def test_a_proposal_of_an_impossible_combination_is_refused_too(client):
    body = client.post("/api/propose",
                       json={"changes": {"rod.shank_flange": 0.020}}).json()
    assert body["ok"] is False
    assert body["violation"]["error"] == "impossible_combination"


def test_the_cycle_is_indexed_by_the_same_slider_as_the_sweep(client):
    """The viewport scrubs the stress field with the crank slider, so the
    two arrays have to be the same length or the field shown belongs to a
    different crank angle than the one on screen."""
    field = client.get("/api/fea/pin?elements=8000").json()
    sweep = client.get("/api/sweep").json()

    assert field["ok"]
    assert len(field["cycle"]["scale"]) == len(sweep["theta_deg"])
    assert field["cycle"]["theta_deg"][0] == pytest.approx(
        sweep["theta_deg"][0], abs=1e-9)


def test_the_field_comes_with_what_to_measure_it_against(client):
    body = client.get("/api/fea/pin?elements=8000").json()
    allowable = body["allowable"]
    assert allowable["allowable_pa"] > 0
    assert allowable["basis"]
    assert allowable["temperature_c"] > 20


def test_the_front_end_is_never_cached(client):
    """The symptom this prevents is vicious: the Python restarts and picks up
    every change, the files on disk are right, and the browser keeps running
    an old app.js -- so all the evidence says the folder was never updated
    when it was. ETag and Last-Modified are not enough, because Chrome will
    reuse a script from its memory cache without revalidating."""
    for path in ("/", "/static/app.js", "/static/styles.css"):
        headers = client.get(path).headers
        assert "no-store" in headers.get("cache-control", "")


def test_the_server_can_say_what_it_is_running(client):
    """So "is it stale?" has an answer instead of being a guess."""
    body = client.get("/api/version").json()
    assert body["ok"]
    assert body["python"]
    assert body["package_dir"].endswith("psrt")
    assert set(body["static"]) == {"index.html", "app.js", "styles.css"}
    assert all(f["sha256"] for f in body["static"].values())
