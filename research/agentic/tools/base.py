"""Tool protocol for the SURDS agentic harness.

Deliberately tiny. DeepEyesV2 ships a 190-line metaclass registry for this; a dict of
callables is enough and does not hide control flow.

Design rules, both learned from auditing the reference repos:

  1. `execute` NEVER raises. A malformed tool call is a training signal, not an
     exception -- the error text goes back to the model as the observation. This is
     Molt's `run_python` contract (`examples/python/tools/python_executor.py:87`), and
     the opposite of DeepEyesV2's RL path, where a sandbox exception returns None and
     silently truncates the episode (`deepeyesv2.py:148-150`), biasing the GRPO group.

  2. Tool state is per-trajectory and explicitly scoped by `ToolContext`. DeepEyesV2
     made `session_id` a class attribute (`deepeyesv2.py:76`), so every concurrent
     rollout shared one Jupyter kernel and overwrote each other's variables. We hold no
     cross-call state at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from ..frames import FrameSpec, frame_spec


@dataclass
class Observation:
    """What a tool hands back to the turn loop.

    `text` is always present; `images` may be empty. Both go into the next user turn.
    `ok` distinguishes a successful call from an error observation -- the model sees
    both, but only `ok=False` counts toward the execution-failure metric and the
    `R_exec_fail` reward penalty.
    """

    text: str
    images: List[Any] = field(default_factory=list)   # PIL.Image.Image
    ok: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def error(cls, message: str, **meta) -> 'Observation':
        return cls(text=f'Error: {message}', images=[], ok=False, meta=meta)


@dataclass
class ToolContext:
    """Per-trajectory, per-sample context handed to every tool call.

    `image` is the NATIVE-resolution frame (1600x900 for SURDS). The python_repl is
    seeded with this, so sandbox coordinates == native coordinates -- see `frames.py`.
    The crop tool converts into FETCHED space itself, because that is the space
    `PIL.Image.crop` must operate in for the result to line up with what the model saw.
    """

    image: Any                                        # PIL.Image.Image, native size
    image_path: str
    spec: FrameSpec = field(default_factory=frame_spec)
    template_type: Optional[str] = None
    intrinsics: Optional[Any] = None                  # 3x3 camera matrix K, if available
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def native_wh(self) -> Tuple[int, int]:
        return self.spec.native_wh


class Tool(Protocol):
    """Structural protocol. Mirrors Molt's `schema` + `execute(arguments)` pair so a
    later port is mechanical."""

    name: str
    schema: Dict[str, Any]

    def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> Observation:
        ...


def build_registry(*tools: Tool) -> Dict[str, Tool]:
    """A dict, not a metaclass."""
    registry: Dict[str, Tool] = {}
    for tool in tools:
        if tool.name in registry:
            raise ValueError(f'duplicate tool name: {tool.name}')
        registry[tool.name] = tool
    return registry
