"""
This module sets up configurations to run agent
It provides different constructs to interact with the LLM
Embedding model = Qwen/Qwen3-Embedding-0.6B
Generative model = Qwen/Qwen2.5-7B-Instruct
"""

import logging
import os
import warnings
from typing import Literal

import dspy
from langchain_core.messages import BaseMessage

CORPUS_PATH = os.getenv("CORPUS_PATH")

PCORPUS_PATH = os.getenv("PCORPUS_PATH")

DB_PATH = os.getenv("DB_PATH")

QUERY = os.getenv("QUERY")

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
        desc="Concise explanation of the decision in 50 words"
    )

# Module to make plan
plan = dspy.Predict(Plan)


class Tune(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The new questions")
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the decision in 50 words"
    )

# Module to tune reflection
tune = dspy.Predict(Tune)

class Decide(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    next_decision: Literal["researcher", "reflector", "end"] = dspy.OutputField(
        desc="The next step to take"
    )
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the decision in 50 words"
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
    input_text: list[BaseMessage] = dspy.InputField()
    existing_history: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="Reformulated query")

rephrase_pred = dspy.Predict(Rephrase)

class Analyze(dspy.Signature):
    context: list[BaseMessage] = dspy.InputField()
    existing_history: list[BaseMessage] = dspy.InputField()
    input_text: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="Analysis (≤200 words)")

analyze_pred = dspy.Predict(Analyze)

class Check(dspy.Signature):
    answer: list[BaseMessage] = dspy.InputField()
    input_text: list[BaseMessage] = dspy.InputField()
    decision: str = dspy.OutputField(desc='"yes" or "no"')

check_pred = dspy.Predict(Check)

class Summarize(dspy.Signature):
    full_context: list[BaseMessage] = dspy.InputField()
    summary: str = dspy.OutputField(desc="Bullet-point summary")

summarize_pred = dspy.Predict(Summarize)

class Synthesize(dspy.Signature):
    input_text: list[BaseMessage] = dspy.InputField()
    updated_history: list[BaseMessage] = dspy.InputField()
    conclusion: str = dspy.OutputField(desc="Final paragraph")

synthesize_pred = dspy.Predict(Synthesize)

class Subquery(dspy.Signature):
    query: list[BaseMessage] = dspy.InputField()
    answer: list[str] = dspy.OutputField(desc="The list of smaller tasks")

subquery = dspy.Predict(Subquery)

class Finalize(dspy.Signature):
    query: list[BaseMessage] = dspy.InputField()
    subqueries: list[BaseMessage] = dspy.InputField()
    answers: list[BaseMessage] = dspy.InputField()
    conclusion: str = dspy.OutputField(desc="Final answer")

finalize_pred = dspy.Predict(Finalize)
