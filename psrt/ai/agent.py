"""The conversation loop: model, tools, design state.

The loop itself is small. The model is given the briefing and the toolbox, it
asks for tools, the tools run against the live Session, the results go back,
and it eventually answers. What matters is what surrounds that loop:

* **A round cap.** A model that loops on tools forever is a runaway bill, so
  the loop stops and says so rather than continuing quietly.
* **Result truncation.** Some tool results are large. They are capped with a
  visible note, so the model knows it is looking at a trimmed result rather
  than a complete one.
* **Proposals are surfaced as events.** A staged proposal is not just text in
  the transcript; it becomes an event the UI renders as an accept/discard
  card, so committing stays a deliberate act by the user.
* **No key is a clear message, not a stack trace.**

The class takes an injected ``client``, which is how the whole loop is tested
without an API key: the tests drive it with a scripted fake model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .briefing import system_prompt
from .tools import TOOLS, dispatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(path: Path = ENV_FILE) -> list:
    """Read a .env file beside the project, if there is one.

    Setting an environment variable on Windows is fiddly enough that people
    give up on it, so a plain file in the project folder is the friendlier
    route. Values already in the environment win, so an explicitly exported
    key always beats the file.

    Returns the names it set, which is what the status endpoint reports so
    you can see the file was actually found.
    """
    if not path.is_file():
        return []
    loaded = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name, value = name.strip(), value.strip().strip("\"'")
            if name and value and name not in os.environ:
                os.environ[name] = value
                loaded.append(name)
    except OSError:
        return []
    return loaded


load_env_file()

DEFAULT_MODEL = os.environ.get("PSRT_MODEL", "claude-sonnet-4-5")
MAX_ROUNDS = 12
MAX_RESULT_CHARS = 24_000
MAX_TOKENS = 4096


class MissingAPIKey(RuntimeError):
    pass


def no_key_message() -> str:
    return (
        "No Anthropic API key found. The easiest fix: copy .env.example to "
        f".env in {PROJECT_ROOT.name}, put your key in it, and restart the "
        "tool. Get a key from console.anthropic.com under API keys. "
        "Everything else in the application works without one -- this only "
        "affects the assistant.")


def _blocks(message):
    return getattr(message, "content", None) or []


def _attr(block, name, default=None):
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


class Agent:
    """One conversation about one design."""

    def __init__(self, session, client=None, model: str = DEFAULT_MODEL,
                 api_key: str | None = None, stream: bool = True):
        self.session = session
        self.model = model
        self.stream = stream
        self.messages: list = []
        self._client = client
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    # -- client -------------------------------------------------------------

    @property
    def client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise MissingAPIKey(no_key_message())
        from anthropic import Anthropic
        self._client = Anthropic(api_key=self._api_key)
        return self._client

    @property
    def available(self) -> bool:
        return self._client is not None or bool(self._api_key)

    # -- the loop -----------------------------------------------------------

    def run(self, user_text: str):
        """Yield events for one user turn. A generator, so the caller can
        stream them straight out to the browser."""
        self.messages.append({"role": "user", "content": user_text})

        try:
            client = self.client
        except MissingAPIKey as exc:
            yield {"type": "error", "message": str(exc)}
            yield {"type": "done"}
            return

        for round_index in range(MAX_ROUNDS):
            try:
                message = yield from self._call(client)
            except Exception as exc:                            # noqa: BLE001
                yield {"type": "error",
                       "message": f"{type(exc).__name__}: {exc}"}
                yield {"type": "done"}
                return

            content = []
            tool_calls = []
            for block in _blocks(message):
                kind = _attr(block, "type")
                if kind == "text":
                    content.append({"type": "text",
                                    "text": _attr(block, "text", "")})
                elif kind == "tool_use":
                    call = {"id": _attr(block, "id"),
                            "name": _attr(block, "name"),
                            "input": _attr(block, "input") or {}}
                    tool_calls.append(call)
                    content.append({"type": "tool_use", **call})
            self.messages.append({"role": "assistant", "content": content})

            if not tool_calls:
                yield {"type": "done"}
                return

            results = []
            for call in tool_calls:
                yield {"type": "tool_use", "name": call["name"],
                       "input": call["input"]}
                result = dispatch(self.session, call["name"], call["input"])

                if call["name"] == "propose" and result.get("ok"):
                    yield {"type": "proposal",
                           "proposal_id": result["proposal_id"],
                           "changes": result["changes"],
                           "rationale": result.get("rationale", ""),
                           "regressions": result.get("regressions", []),
                           "deltas": result.get("deltas", {})}
                elif call["name"] in ("commit", "revert", "set_speed"):
                    yield {"type": "design_changed", "tool": call["name"],
                           "ok": bool(result.get("ok"))}

                yield {"type": "tool_result", "name": call["name"],
                       "ok": "error" not in result and result.get("ok", True)}
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": _serialise(result),
                })
            self.messages.append({"role": "user", "content": results})

        yield {"type": "error",
               "message": (f"Stopped after {MAX_ROUNDS} rounds of tool calls "
                           "without reaching an answer. Ask something "
                           "narrower, or check whether a tool is failing.")}
        yield {"type": "done"}

    def _call(self, client):
        """One model call. Yields text as it arrives; returns the message."""
        request = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system_prompt(self.session),
            "tools": TOOLS,
            # A snapshot, not the live list: the loop keeps appending to
            # self.messages, and a client holding a reference would see turns
            # that had not happened when the request was made.
            "messages": list(self.messages),
        }

        streamer = getattr(getattr(client, "messages", None), "stream", None)
        if self.stream and callable(streamer):
            with streamer(**request) as stream:
                for chunk in stream.text_stream:
                    yield {"type": "text", "text": chunk}
                return stream.get_final_message()

        message = client.messages.create(**request)
        for block in _blocks(message):
            if _attr(block, "type") == "text":
                yield {"type": "text", "text": _attr(block, "text", "")}
        return message


def _serialise(result: dict) -> str:
    text = json.dumps(result, default=str)
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return (text[:MAX_RESULT_CHARS]
            + f"\n\n[truncated at {MAX_RESULT_CHARS} characters. Ask for a "
              "narrower slice -- a single component, a shorter rpm range -- "
              "rather than assuming the rest looks like this.]")
