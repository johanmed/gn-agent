"""
This modules creates an adapter that wraps existing GNAgent instance and exposes a serializable config for GEPA
"""

import asyncio
import copy
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import dspy
from langchain_core.messages import SystemMessage

from all_config import *

PROMPT_NAMES = [
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


class PromptSig(dspy.Signature):
    query: str = dspy.InputField()
    prompt_output: str = dspy.OutputField()


class GNAgentFullSig(dspy.Signature):
    query: str = dspy.InputField()
    answer: str = dspy.OutputField()
    reasoning: str = dspy.OutputField()


class GNAgentAdapter(dspy.Module):
    def __init__(self, agent_config):
        super().__init__()
        self.config = agent_config
        self.executor = ThreadPoolExecutor(max_workers=1)

        self._predictors = {}
        for name in PROMPT_NAMES:
            pred = dspy.Predict(PromptSig)
            pred.prompt_text = agent_config["prompts"].get(name, "")
            self._predictors[name] = pred
            setattr(self, name, pred)

        self.full = dspy.Predict(GNAgentFullSig)

    def _build_agent(self):
        base = {k: v for k, v in self.config.items() if k != "prompts"}
        valid_keys = {*PROMPT_NAMES, "corpus_path", "pcorpus_path", "db_path"}
        base = {k: v for k, v in base.items() if k in valid_keys}
        for k in {"corpus_path", "pcorpus_path", "db_path"}:
            if k in base and isinstance(base[k], str):
                base[k] = Path(base[k])

        prompts = {}
        for name in PROMPT_NAMES:
            pred = self._predictors[name]
            text = getattr(pred, "prompt_text", "")
            prompts[name] = SystemMessage(content=text)

        if hasattr(self, "_embedder"):
            base["embedder"] = self._embedder
        return GNAgent(**base, **prompts)

    @staticmethod
    def _run_handler(agent, query: str):
        return asyncio.run(agent.handler(query))

    def forward(self, query: str) -> dspy.Prediction:
        agent = self._build_agent()
        answer, reasoning = self.executor.submit(
            self._run_handler, agent, query
        ).result()
        logging.info(f"System feedback: {answer}")
        return self.full(query=query, answer=str(answer), reasoning=str(reasoning))

    def __call__(self, *args, **kwargs):
        if args and isinstance(args[0], dspy.Example):
            return self.forward(query=args[0].query)
        return self.forward(**kwargs)

    def predictors(self):
        return list(self._predictors.values())

    def named_predictors(self):
        return [(n, p) for n, p in self._predictors.items()]

    def reset_copy(self):
        new = self.__class__.__new__(self.__class__)
        new.config = copy.deepcopy(self.config)
        new._predictors = {}
        for name, pred in self._predictors.items():
            new_pred = copy.deepcopy(pred)
            new._predictors[name] = new_pred
            setattr(new, name, new_pred)
        new.executor = ThreadPoolExecutor(max_workers=1)
        new.full = copy.deepcopy(self.full)
        return new


def extract_config(agent) -> Dict[str, Any]:
    """Convert GNAgent to JSON-serialisable dict"""
    config: Dict[str, Any] = {}
    prompt_names = set(PROMPT_NAMES)

    allowed_fields = {"corpus_path", "pcorpus_path", "db_path"}
    for k, v in agent.__dict__.items():
        if k not in allowed_fields or k in prompt_names:
            continue
        config[k] = str(v) if isinstance(v, Path) else v

    config["prompts"] = {}
    for name in PROMPT_NAMES:
        obj = getattr(agent, name, None)
        if obj is not None:
            if isinstance(obj, SystemMessage):
                config["prompts"][name] = obj.content
            else:
                config["prompts"][name] = str(obj)

    return config


agent = GNAgent(
    corpus_path=CORPUS_PATH,
    pcorpus_path=PCORPUS_PATH,
    db_path=DB_PATH,
    naturalize_prompt=naturalize_prompt,
    rephrase_prompt=rephrase_prompt,
    analyze_prompt=analyze_prompt,
    check_prompt=check_prompt,
    summarize_prompt=summarize_prompt,
    synthesize_prompt=synthesize_prompt,
    split_prompt=split_prompt,
    finalize_prompt=finalize_prompt,
    sup_prompt1=sup_prompt1,
    sup_prompt2=sup_prompt2,
    plan_prompt=plan_prompt,
    refl_prompt=refl_prompt,
)
config = extract_config(agent)
