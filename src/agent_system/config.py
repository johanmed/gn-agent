"""
Setup and LLMs
1. Embedding model = Qwen/Qwen3-Embedding-0.6B
2. Generative model = Qwen/Qwen2.5-7B-Instruct
"""

import logging
import warnings
from typing import Literal

import dspy
from langchain_core.messages import BaseMessage

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(
    filename="log_agent.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


# X: Remove hard-coded path.
CORPUS_PATH = "/home/johannesm/all_corpus/"

# X: Remove hard_coded path.
PCORPUS_PATH = "/home/johannesm/all_tmp/full_docs.txt"

# X: Remove hard-coded path.
DB_PATH = "/home/johannesm/all_tmp/full_chroma_db"


EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

GENERATIVE_MODEL = dspy.LM(
    model="openai/Qwen/Qwen2.5-7B-Instruct",
    api_base="http://localhost:7501/v1",
    api_key="local",
    model_type="chat",
    max_tokens=10_000,
    n_ctx=30_000,
    seed=2_025,
    temperature=0,
    verbose=False,
)

dspy.configure(lm=GENERATIVE_MODEL)

generate = dspy.ChainOfThought("question -> answer: str")


class Subquery(dspy.Signature):
    query: str = dspy.InputField(desc="the task to solve")
    answer: list = dspy.OutputField(
        desc="the smaller tasks that help solve the main task"
    )
    reasoning: str = dspy.OutputField(
        desc="provide a concise explanation of the thought process for the input, limited to approximately 50 words."
    )


subquery = dspy.Predict(Subquery)


class Decide(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField(
        desc="the background to use to make decision"
    )
    next: Literal["researcher", "reflector", "end"] = dspy.OutputField(
        desc="the next step to take"
    )
    reasoning: str = dspy.OutputField(
        desc="provide a concise explanation of the decision given the input, limited to approximately 50 words."
    )


supervise = dspy.Predict(Decide)


class Plan(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField(
        desc="the background to use to decide"
    )
    answer: str = dspy.OutputField(desc="the task result")
    reasoning: str = dspy.OutputField(
        desc="provide a concise explanation of the thought process for the input, limited to approximately 50 words."
    )


plan = dspy.Predict(Plan)


class Tune(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField(
        desc="the background to use to ask new questions"
    )
    answer: str = dspy.OutputField(desc="the new questions")
    reasoning: str = dspy.OutputField(
        desc="provide a concise explanation of the thought process for the input, limited to approximately 50 words."
    )


tune = dspy.Predict(Tune)
