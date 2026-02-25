"""This modules creates an adapter that wraps existing GNAgent instance and exposes it as a dspy module for GEPA"""

import asyncio
import copy
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

import dspy
from gnagent.agent import GNAgent

from config2 import *

warnings.filterwarnings("ignore")


class GNAgentFullSig(dspy.Signature):
    query: str = dspy.InputField()
    answer: str = dspy.OutputField()


class GNAgentAdapter(dspy.Module):
    def __init__(self, agent_config):
        super().__init__()
        self.config = agent_config
        self.executor = ThreadPoolExecutor(max_workers=1)
        pred = dspy.Predict(GNAgentFullSig)
        self.general = pred
        pred.prompt = "You are performing a genomic task. Make sure to explore as many possibilities as possible."
        self._predictors = {"general_prompt": pred}

    def _build_agent(self):
        base = {k: v for k, v in self.config.items()}
        valid_keys = {
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
            "expert_prompt",
            "corpus_path",
            "pcorpus_path",
            "db_path",
            "ext_db_path",
        }
        base = {k: v for k, v in base.items() if k in valid_keys}
        return GNAgent(**base)

    @staticmethod
    def _run_handler(agent, query: str):
        return asyncio.run(agent.handler(query))

    def forward(self, query: str) -> dspy.Prediction:
        agent = self._build_agent()
        answer = self.executor.submit(self._run_handler, agent, query).result()
        return self.general(query=query, answer=str(answer))

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
        return new


def extract_config(agent) -> Dict[str, Any]:
    """Convert GNAgent to JSON-serialisable dict"""
    config: Dict[str, Any] = {}
    allowed_fields = {
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
        "expert_prompt",
        "corpus_path",
        "pcorpus_path",
        "db_path",
        "ext_db_path",
    }
    for k, v in agent.__dict__.items():
        if k not in allowed_fields:
            continue
        config[k] = v

    return config


agent = GNAgent(
    corpus_path=CORPUS_PATH,
    pcorpus_path=PCORPUS_PATH,
    db_path=DB_PATH,
    ext_db_path=EXT_DB_PATH,
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
    expert_prompt=expert_prompt,
)
config = extract_config(agent)

dspy_agent = GNAgentAdapter(config)
