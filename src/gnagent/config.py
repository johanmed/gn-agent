"""
This module sets up configurations to run agent
It provides different constructs to interact with the LLM
Embedding model = Qwen/Qwen3-Embedding-0.6B
Generative model = Qwen/Qwen2.5-7B-Instruct
Note: Need to customize paths
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

# TODO: Customize path
CORPUS_PATH = "/home/johannesm/all_corpus/"

# TODO: Customize path
PCORPUS_PATH = "/home/johannesm/all_tmp/new_docs.txt"

# TODO: Customize path
DB_PATH = "/home/johannesm/all_tmp/new_chroma_db"

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

GENERATIVE_MODEL = dspy.LM(
    model="openai/Qwen/Qwen2.5-7B-Instruct",  # should match shell config
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


class Plan(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The plan to solve the task")
    reasoning: str = dspy.OutputField(
        desc="Provide a concise explanation of the thought process for the input, limited to approximately 50 words."
    )
# Module to make plan
plan = dspy.Predict(Plan)


class Tune(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The new questions")
    reasoning: str = dspy.OutputField(
        desc="Provide a concise explanation of the thought process for the input, limited to approximately 50 words."
    )
# Module to tune reflection
tune = dspy.Predict(Tune)


class Decide(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    next: Literal["researcher", "reflector", "end"] = dspy.OutputField(
        desc="The next step to take"
    )
    reasoning: str = dspy.OutputField(
        desc="Provide a concise explanation of the decision given the input, limited to approximately 50 words."
    )
# Module to manage system
supervise = dspy.Predict(Decide)


class End(dspy.Signature):
    question: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="Well formulated final feedback")
# Module to wrap up
end = dspy.Predict(End)

# Specialized modules for researcher

class Naturalize(dspy.Signature):
    text: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="Natural English sentence")
naturalize_pred = dspy.Predict(Naturalize)

class Rephrase(dspy.Signature):
    input: list[BaseMessage] = dspy.InputField()
    existing_history: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="Reformulated query")
rephrase_pred = dspy.Predict(Rephrase)

class Analyze(dspy.Signature):
    context: list[BaseMessage] = dspy.InputField()
    existing_history: list[BaseMessage] = dspy.InputField()
    input: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="Analysis (≤200 words)")
analyze_pred = dspy.Predict(Analyze)

class Check(dspy.Signature):
    answer: list[BaseMessage] = dspy.InputField()
    input: list[BaseMessage] = dspy.InputField()
    decision: str = dspy.OutputField(desc='"yes" or "no"')
check_pred = dspy.Predict(Check)

class Summarize(dspy.Signature):
    full_context: list[BaseMessage] = dspy.InputField()
    summary: str = dspy.OutputField(desc="Bullet-point summary")
summarize_pred = dspy.Predict(Summarize)

class Synthesize(dspy.Signature):
    input: list[BaseMessage] = dspy.InputField()
    updated_history: list[BaseMessage] = dspy.InputField()
    conclusion: str = dspy.OutputField(desc="Final paragraph")
synthesize_pred = dspy.Predict(Synthesize)

class Subquery(dspy.Signature):
    query: list[BaseMessage] = dspy.InputField()
    answer: list[str] = dspy.OutputField(
        desc="The list of smaller tasks"
     )
subquery = dspy.Predict(Subquery)

class Finalize(dspy.Signature):
    query: list[BaseMessage] = dspy.InputField()
    subqueries: list[BaseMessage] = dspy.InputField()
    answers: list[BaseMessage] = dspy.InputField()
    conclusion: str = dspy.OutputField(desc="Final answer")
finalize_pred = dspy.Predict(Finalize)
