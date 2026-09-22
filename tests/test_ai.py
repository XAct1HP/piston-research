"""The AI layer: the toolbox, its guardrails, and the conversation loop.

No API key is needed to run any of this. The tools are pure functions over a
Session, and the agent takes an injected client, so the whole loop is driven
here by a scripted fake model. That is deliberate: the tools are the valuable
part and they should be testable without a network call.
"""

import json
import os

import pytest

from psrt.ai import TOOLS, dispatch, tool_names
from psrt.ai.agent import Agent, MissingAPIKey
from psrt.ai.briefing import BRIEFING, context_block, system_prompt
from psrt.sensitivity import OBJECTIVES, rank_levers
from psrt.server.session import Session

HERE = os.path.dirname(os.path.abspath(__file__))
LS3 = os.path.join(os.path.dirname(HERE), "examples", "ls3.json")


@pytest.fixture
def session():
    return Session.from_file(LS3)


# --- the schemas -----------------------------------------------------------

def test_every_tool_has_a_well_formed_schema():
    for tool in TOOLS:
        assert tool["name"] and tool["description"], tool
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert isinstance(schema.get("properties", {}), dict)
        for name in schema.get("required", []):
            assert name in schema["properties"], (tool["name"], name)


def test_tool_names_are_unique():
    assert len(tool_names()) == len(set(tool_names()))


def test_every_tool_dispatches_and_returns_json(session):
    calls = {
        "get_design_state": {"section": "engine"},
        "get_metrics": {},
        "get_margins": {},
        "explain_margin": {"query": "rod"},
        "get_curve": {"start_rpm": 3000, "stop_rpm": 5000, "step_rpm": 1000},
        "rank_levers": {"objective": "torque", "paths": ["engine.bore"]},
        "propose": {"changes": {"piston.crown_thickness": 0.0066},
                    "rationale": "test"},
        "set_speed": {"rpm": 5000},
        "revert": {},          # after set_speed, so there is something to undo
        "check_geometry": {},
        "compare_bore_profiles": {},
    }
    for name, args in calls.items():
        result = dispatch(session, name, args)
        assert isinstance(result, dict), name
        json.dumps(result)                        # must be serialisable
        assert "error" not in result, (name, result.get("error"))


def test_results_never_contain_nan_or_infinity(session):
    for name in ("get_metrics", "get_margins", "check_geometry"):
        assert "NaN" not in json.dumps(dispatch(session, name, {}))
        assert "Infinity" not in json.dumps(dispatch(session, name, {}))


# --- refusals are answers --------------------------------------------------

def test_a_locked_write_comes_back_as_a_reason(session):
    result = dispatch(session, "propose", {
        "changes": {"block.as_built_bore": 0.11}, "rationale": "nope"})
    assert result["ok"] is False
    assert "the block exists" in result["violation"]["reason"]


def test_an_out_of_bounds_write_names_its_bound(session):
    """The bound quoted back must be the one the block envelope set, not a
    number pinned in this test -- otherwise the test stops tracking the
    model the moment the envelope learns something new."""
    from psrt.envelope import envelope

    result = dispatch(session, "propose", {
        "changes": {"engine.bore": 0.12}, "rationale": "nope"})
    assert result["ok"] is False
    assert result["violation"]["bound"] == pytest.approx(
        envelope(session.state).max_bore, abs=1e-9)


def test_millimetres_are_caught_before_the_bounds_check(session):
    """'above the upper bound of 0.10376' is true but unhelpful when the real
    mistake is units."""
    result = dispatch(session, "propose", {
        "changes": {"engine.bore": 103.5}, "rationale": "wrong units"})
    assert result["ok"] is False
    assert result["violation"]["error"] == "unit_mistake"
    assert "METRES" in result["violation"]["reason"]


def test_a_sensible_metre_value_is_not_mistaken_for_millimetres(session):
    result = dispatch(session, "propose", {
        "changes": {"engine.bore": 0.1035}, "rationale": "real overbore"})
    assert result["ok"] is True


def test_unknown_tool_lists_the_real_ones(session):
    error = dispatch(session, "no_such_tool", {})["error"]
    assert "unknown tool" in error and "get_metrics" in error


def test_unknown_section_and_component_are_explained(session):
    assert "sections are" in dispatch(
        session, "get_design_state", {"section": "flywheel"})["error"]
    assert "try:" in dispatch(
        session, "get_margins", {"component": "camshaft"})["error"]


def test_a_silly_curve_request_is_refused(session):
    assert "error" in dispatch(session, "get_curve",
                               {"start_rpm": 6000, "stop_rpm": 1000})
    assert "80 points" in dispatch(session, "get_curve", {
        "start_rpm": 1000, "stop_rpm": 9000, "step_rpm": 10})["error"]


def test_speed_outside_the_plausible_range_is_refused(session):
    assert "error" in dispatch(session, "set_speed", {"rpm": 99000})


# --- the staging property --------------------------------------------------

def test_a_proposal_leaves_the_live_design_untouched(session):
    before = session.state.fingerprint()
    result = dispatch(session, "propose", {
        "changes": {"engine.bore": 0.1035}, "rationale": "overbore"})
    assert result["ok"] is True
    assert session.state.fingerprint() == before


def test_committing_applies_it_and_revert_undoes_it(session):
    before = session.state.fingerprint()
    proposal = dispatch(session, "propose", {
        "changes": {"piston.crown_thickness": 0.0070},
        "rationale": "thicker crown"})
    dispatch(session, "commit", {"proposal_id": proposal["proposal_id"]})
    assert session.state.fingerprint() != before
    dispatch(session, "revert", {})
    assert session.state.fingerprint() == before


def test_a_proposal_always_reports_what_got_worse(session):
    result = dispatch(session, "propose", {
        "changes": {"engine.bore": 0.1035}, "rationale": "overbore"})
    names = [r["metric"] for r in result["regressions"]]
    assert any("side_thrust" in n for n in names)


def test_explain_margin_carries_the_working(session):
    match = dispatch(session, "explain_margin", {"query": "crown"})["matches"][0]
    assert match["equation"] and match["reference"] and match["condition"]
    assert match["inputs"], "the numbers that went in must be there"
    assert match["notes"], "the caveats must be there"


# --- sensitivity -----------------------------------------------------------

def test_bore_appears_among_the_torque_levers(session):
    """It went missing once: the probe step was larger than the bore's entire
    remaining headroom, so both probes were refused and the most important
    lever in the engine vanished from the ranking."""
    result = rank_levers(session.state, "torque",
                         paths=["engine.bore", "engine.compression_ratio"])
    paths = [l["path"] for l in result["levers"]]
    assert "engine.bore" in paths


def test_bore_is_limited_by_its_declared_bound(session):
    lever = next(l for l in rank_levers(
        session.state, "torque", paths=["engine.bore"])["levers"]
        if l["path"] == "engine.bore")
    assert lever["direction"] == "increase"
    assert lever["limited_by"] == "parameter bound"
    # Whatever the binding limit happens to be, the lever has to name it
    # rather than report a bare number.
    assert lever["limit_reason"]
    assert lever["limit_reason"].split(":")[0] in {
        limit.name for limit in __import__(
            "psrt.envelope", fromlist=["envelope"]
        ).envelope(session.state).known_limits}
    assert 0 < lever["reachable_gain_percent"] < 5


def test_levers_report_their_side_effects(session):
    lever = next(l for l in rank_levers(
        session.state, "torque", paths=["engine.bore"])["levers"]
        if l["path"] == "engine.bore")
    assert any(e.get("worse") for e in lever["side_effects"])


def test_a_locked_parameter_is_never_offered_as_a_lever(session):
    session.state.lock("engine.stroke", "no new crankshaft")
    paths = [l["path"] for l in rank_levers(
        session.state, "torque",
        paths=["engine.stroke", "engine.bore"])["levers"]]
    assert "engine.stroke" not in paths


def test_ranking_is_sorted_by_what_is_reachable(session):
    levers = rank_levers(session.state, "torque")["levers"]
    gains = [abs(l["reachable_gain_percent"] or 0) for l in levers]
    assert gains == sorted(gains, reverse=True)


def test_every_shorthand_objective_resolves(session):
    for objective in OBJECTIVES:
        result = rank_levers(session.state, objective,
                             paths=["engine.bore"], find_limits=False)
        assert result["baseline"] is not None, objective


def test_an_unknown_objective_says_what_is_available(session):
    with pytest.raises(ValueError, match="torque"):
        rank_levers(session.state, "vibes")


# --- the briefing ----------------------------------------------------------

def test_the_briefing_carries_the_rules_that_matter():
    for phrase in ("Call a tool for every number",
                   "propose", "what got WORSE",
                   "POSITIVE IN COMPRESSION", "overlap TDC",
                   "calibrated"):
        assert phrase in BRIEFING, phrase


def test_the_briefing_names_the_calibrated_parameters():
    for name in ("crown_support_radius_fraction", "support_span_factor",
                 "Volumetric efficiency"):
        assert name in BRIEFING


def test_the_context_block_describes_the_live_design(session):
    block = context_block(session)
    assert "GM LS3" in block
    assert "Minimum safety factor" in block
    assert "Locked" in block
    assert "block.as_built_bore" in block


def test_the_system_prompt_is_briefing_plus_context(session):
    prompt = system_prompt(session)
    assert prompt.startswith(BRIEFING[:40])
    assert "103.25" in prompt


# --- the agent loop --------------------------------------------------------

class Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class ScriptedMessages:
    """A fake model: a list of replies, handed out in order."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.script[min(len(self.requests) - 1, len(self.script) - 1)]
        return Block(content=reply, stop_reason="end_turn")


class ScriptedClient:
    def __init__(self, script):
        self.messages = ScriptedMessages(script)


def agent_for(session, script):
    return Agent(session, client=ScriptedClient(script))


def test_the_loop_runs_a_tool_and_comes_back_with_an_answer(session):
    agent = agent_for(session, [
        [Block(type="text", text="Looking."),
         Block(type="tool_use", id="a", name="get_metrics", input={})],
        [Block(type="text", text="Pin ovalisation is binding.")],
    ])
    events = list(agent.run("what is limiting this?"))
    kinds = [e["type"] for e in events]
    assert "tool_use" in kinds and "tool_result" in kinds
    assert kinds[-1] == "done"
    assert "".join(e["text"] for e in events if e["type"] == "text").endswith(
        "Pin ovalisation is binding.")


def test_a_proposal_becomes_an_event_the_ui_can_render(session):
    agent = agent_for(session, [
        [Block(type="tool_use", id="a", name="propose",
               input={"changes": {"piston.crown_thickness": 0.0070},
                      "rationale": "relieve crown fatigue"})],
        [Block(type="text", text="Staged.")],
    ])
    proposals = [e for e in agent.run("thicken the crown")
                 if e["type"] == "proposal"]
    assert len(proposals) == 1
    assert proposals[0]["proposal_id"]
    assert proposals[0]["rationale"] == "relieve crown fatigue"


def test_the_loop_does_not_commit_on_its_own(session):
    before = session.state.fingerprint()
    agent = agent_for(session, [
        [Block(type="tool_use", id="a", name="propose",
               input={"changes": {"engine.bore": 0.1035},
                      "rationale": "overbore"})],
        [Block(type="text", text="Proposed, waiting on you.")],
    ])
    list(agent.run("bore it out"))
    assert session.state.fingerprint() == before


def test_a_runaway_tool_loop_is_capped(session):
    """A model that keeps calling tools forever is a runaway bill."""
    agent = agent_for(session, [
        [Block(type="tool_use", id="a", name="get_metrics", input={})],
    ])
    events = list(agent.run("loop forever"))
    errors = [e for e in events if e["type"] == "error"]
    assert errors and "rounds of tool calls" in errors[0]["message"]
    assert events[-1]["type"] == "done"


def test_a_failing_tool_is_reported_not_swallowed(session):
    agent = agent_for(session, [
        [Block(type="tool_use", id="a", name="get_margins",
               input={"component": "flywheel"})],
        [Block(type="text", text="No such component.")],
    ])
    results = [e for e in agent.run("check the flywheel")
               if e["type"] == "tool_result"]
    assert results and results[0]["ok"] is False


def test_the_tool_result_reaches_the_model(session):
    agent = agent_for(session, [
        [Block(type="tool_use", id="a", name="get_metrics", input={})],
        [Block(type="text", text="done")],
    ])
    list(agent.run("metrics please"))
    second = agent._client.messages.requests[1]
    payload = second["messages"][-1]["content"][0]
    assert payload["type"] == "tool_result"
    assert "brake_torque_nm" in payload["content"]


def test_the_model_is_given_the_briefing_and_the_tools(session):
    agent = agent_for(session, [[Block(type="text", text="hello")]])
    list(agent.run("hello"))
    request = agent._client.messages.requests[0]
    assert "Call a tool for every number" in request["system"]
    assert "GM LS3" in request["system"]
    assert len(request["tools"]) == len(TOOLS)


def test_a_huge_tool_result_is_truncated_with_a_note(session):
    from psrt.ai.agent import MAX_RESULT_CHARS, _serialise
    blob = _serialise({"rows": ["x" * 50 for _ in range(2000)]})
    assert len(blob) < MAX_RESULT_CHARS + 400
    assert "truncated" in blob


def test_no_api_key_is_a_message_not_a_crash(session, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = Agent(session)
    events = list(agent.run("hello"))
    assert events[0]["type"] == "error"
    message = events[0]["message"]
    assert ".env" in message, "it should name the easiest fix"
    assert "console.anthropic.com" in message, "and where to get a key"
    assert "works without one" in message, "and that nothing else is blocked"
    assert events[-1]["type"] == "done"
    assert agent.available is False


def test_a_model_error_is_surfaced_not_swallowed(session):
    class Broken:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("model overloaded")
    agent = Agent(session, client=Broken())
    errors = [e for e in agent.run("hi") if e["type"] == "error"]
    assert "model overloaded" in errors[0]["message"]


def test_a_dotenv_file_is_read_when_present(tmp_path, monkeypatch):
    """Setting an environment variable on Windows is fiddly enough that people
    give up on it, so a plain file beside the project has to work."""
    from psrt.ai.agent import load_env_file

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PSRT_MODEL", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n\n"
        'ANTHROPIC_API_KEY="sk-ant-from-file"\n'
        "PSRT_MODEL=some-model\n"
        "a malformed line with no equals sign\n", encoding="utf-8")

    assert set(load_env_file(env)) == {"ANTHROPIC_API_KEY", "PSRT_MODEL"}
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-file"
    assert os.environ["PSRT_MODEL"] == "some-model"


def test_the_environment_wins_over_the_file(tmp_path, monkeypatch):
    from psrt.ai.agent import load_env_file

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-environment")
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n", encoding="utf-8")
    load_env_file(env)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-environment"


def test_a_missing_dotenv_file_is_not_an_error(tmp_path):
    from psrt.ai.agent import load_env_file
    assert load_env_file(tmp_path / "nothing-here") == []


def test_the_key_is_never_reported_back_over_the_api(monkeypatch):
    """The status endpoint says WHERE the key came from, never what it is."""
    from fastapi.testclient import TestClient

    from psrt.server.app import create_app

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    body = TestClient(create_app(LS3)).get("/api/ai/status").text
    assert "sk-ant-secret-value" not in body
    assert "secret" not in body


def test_the_assistant_can_read_the_block_envelope():
    """The assistant must be able to see the block's limits before it
    suggests a bore change, and must see the confidence tier with them."""
    from psrt.ai.tools import dispatch
    from psrt.schema import load_state
    from psrt.server.session import Session

    state, _ = load_state("examples/ls3.json")
    result = dispatch(Session(state), "block_envelope", {})
    assert "error" not in result
    assert result["binding"]["name"]
    assert result["confidence_tier"] in (1, 2, 3)


def test_a_refused_overbore_reaches_the_model_as_an_answer_not_an_error():
    """The refusal carries the physical reason, which is the useful part.
    Surfacing it as a tool error would throw that away."""
    from psrt.ai.tools import dispatch
    from psrt.schema import load_state
    from psrt.server.session import Session

    state, _ = load_state("examples/ls3.json")
    result = dispatch(Session(state), "overbore_cascade", {"bore_mm": 110.0})
    assert result.get("refused") is True
    assert "error" not in result
    assert "as-built" in result["reason"] or "bore" in result["reason"]


def test_the_assistant_gets_the_doubts_with_the_answer():
    """The warnings are the most important part of an optimiser result. If
    they did not reach the model they could not reach the user."""
    from psrt.ai.tools import dispatch
    from psrt.schema import load_state
    from psrt.server.session import Session

    state, _ = load_state("examples/ls3.json")
    result = dispatch(Session(state), "optimise",
                      {"objective": "torque", "levers": 4})
    assert "error" not in result
    assert result["warnings"]
    assert "state" not in result, "the design object is not JSON for a model"
