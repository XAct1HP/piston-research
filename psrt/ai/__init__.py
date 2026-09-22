"""The AI layer: a toolbox, an engineering briefing, and a conversation loop.

The model does not simulate anything and does not calculate anything. It reads
metrics through tools, reasons about which lever to pull, and writes
parameters through tools. All the physics stays in Python where it is tested.

That separation is the whole design. It is the difference between an
engineering tool with a conversational interface and a chatbot that makes up
numbers about pistons.
"""

from .tools import TOOLS, dispatch, tool_names

__all__ = ["TOOLS", "dispatch", "tool_names"]
