"""
This modules creates an adapter that wraps an existing GNAgent instance and exposes a serializable config for GEPA.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class GNAgentAdapter:
    """
    Wraps a GNAgent instance and exposes a serializable config for GEPA.

    - prompt_names: list of attribute names on the agent that contain prompt objects
    - tunable_names: dict of tunable name -> default value (adapter will read agent attributes if present)
    - text_from_prompt: optional callable(prompt_obj) -> str
    - prompt_from_text: optional callable(name, text) -> prompt_obj required for GEPA to apply prompt edits
    """

    def __init__(
        self,
        agent: Any,
        prompt_names: Optional[Iterable[str]] = None,
        tunable_names: Optional[Dict[str, Any]] = None,
        text_from_prompt: Optional[Callable[[Any], str]] = None,
        prompt_from_text: Optional[Callable[[str, str], Any]] = None,
    ):
        self.agent = agent
        if self.agent is None:
            raise ValueError("GNAgentAdapter requires a GNAgent instance")

        self.prompt_names = [
            "naturalize_prompt",
            "rephrase_prompt",
            "analyze_prompt",
            "check_prompt",
            "summarize_prompt",
            "synthesize_prompt",
            "split_prompt",
            "finalize_prompt",
            "sup_prompt1",
            "sup_prompt2",
            "plan_prompt",
            "refl_prompt",
        ]

        default_tunables = {
            "retriever_k": 10,
            "bm25_k": 10,
            "ensemble_weights": [0.5, 0.5],
            "chroma_batch_size": 64,
            "doc_chunk_size": 1,
            "concurrency_limit": 5,
            "threadpool_workers": max(2, (__import__("os").cpu_count() or 1) * 2),
        }
        self.tunable_names = list(default_tunables.keys())
        self.tunable_defaults = default_tunables

        self.text_from_prompt = self._default_text_from_prompt
        self.prompt_from_text = self._default_prompt_from_text

        self._last_config: Dict[str, Any] = {}

    def _default_text_from_prompt(self, prompt_obj: Any) -> str:
        """
        Convert an agent prompt object (or list of prompt objects) to a plain string.
        """
        if prompt_obj is None:
            return ""

        # For list or tuple
        if isinstance(prompt_obj, (list, tuple)):
            parts = []
            for p in prompt_obj:
                content = getattr(p, "content", None)
                if content is None:
                    # maybe p has .text or is a string-like
                    content = getattr(p, "text", str(p))
                parts.append(str(content))
            return "\n".join(parts)

        # For single message object
        content = getattr(prompt_obj, "content", None)
        if content is None:
            content = getattr(prompt_obj, "text", str(prompt_obj))
        return str(content)

    def _default_prompt_from_text(self, name: str, text: str) -> Any:
        """
        Convert a plain string back into a prompt object expected by GNAgent.
        """
        return SystemMessage(text)

    def _read_tunable(self, name: str) -> Any:
        if hasattr(self.agent, name):
            return getattr(self.agent, name)
        return self.tunable_defaults.get(name)

    def get_config(self) -> Dict[str, Any]:
        """
        Public adapter API
        Returns a dict:
        {
          "prompts": { name: text, ... },
          "tunables": { name: value, ... }
        }
        """
        config: Dict[str, Any] = {}
        prompts = {
            name: self.text_from_prompt(getattr(self.agent, name, None))
            for name in self.prompt_names
        }
        config["prompts"] = prompts
        tunables = {name: self._read_tunable(name) for name in self.tunable_names}
        config["tunables"] = tunables
        return config

    def save_config(self, path: str) -> None:
        """
        Save last-applied config snapshot to JSON
        """
        p = Path(path)
        p.write_text(json.dumps(self._last_config, indent=2), encoding="utf-8")
