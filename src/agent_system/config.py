"""
Setup and LLMs

1. Embedding model = Qwen/Qwen3-Embedding-0.6B
2. Generative model = Qwen/Qwen2.5-0.5B
"""

import logging
import warnings
from typing import Literal

import dspy
from langchain_core.messages import BaseMessage

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(
    filename="log_langgraph.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


# X: Remove hard-coded path.
CORPUS_PATH = "/home/johannesm/corpus/"

# X: Remove hard_coded path.
PCORPUS_PATH = "/home/johannesm/tmp/docs.txt"

# X: Remove hard-coded path.
DB_PATH = "/home/johannesm/tmp/chroma_db"


EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

GENERATIVE_MODEL = dspy.LM(
    model="openai/Qwen/Qwen2.5-0.5B",
    api_base="http://localhost:7501/v1",
    api_key="local",
    model_type="chat",
    max_tokens=10_000,
    n_ctx=32_768,
    seed=2_025,
    temperature=0,
    verbose=False,
)


generate = dspy.ChainOfThought("question -> answer: str", lm=GENERATIVE_MODEL)


class SupervisorDecision(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    next: Literal["researcher", "planner", "reflector", "end"] = dspy.OutputField()
    reasoning: str = dspy.OutputField()


supervise = dspy.Predict(SupervisorDecision, lm=GENERATIVE_MODEL)


class Process(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField()
    reasoning: str = dspy.OutputField()


process = dspy.Predict(Process, lm=GENERATIVE_MODEL)
