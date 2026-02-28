"""This module defines constructs for the user prompt optimizer"""

import dspy
from gnagent.config import *
from gnagent.prompts import *

API_KEY = os.getenv("API_KEY")
if API_KEY is None:
    raise ValueError(
        "Valid API_KEY must be specified to use the proprietary model for reflection"
    )

REFLECTION_MODEL = dspy.LM(
    "anthropic/claude-sonnet-4-5-20250929",
    api_key=API_KEY,
    max_tokens=10_000,
    temperature=0,
    verbose=False,
)


class Assessment(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    score: float = dspy.OutputField(
        desc="float between 0 and 1 indicating confidence in statement"
    )


assess = dspy.Predict(Assessment)


def score_answer(
    query: str,
    response: str,
) -> float:

    context = f"User query: {query}\nAI answer: {response}"

    relevance_question = f"{context}\nHow relevant is the answer to the user query?"
    relevance_feedback = assess(question=relevance_question)

    accuracy_question = f"{context}\nHow accurate was AI in addressing the user query?"
    accuracy_feedback = assess(question=accuracy_question)

    score = (relevance_feedback.get("score") + accuracy_feedback.get("score")) / 2

    return score


def assesser(
    gold: dspy.Example,
    pred=None,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> int:
    query = gold["query"]
    response = gold["answer"]
    score = score_answer(query, response)
    return 1 if score >= 0.95 else 0


def smart_assesser(
    gold: dspy.Example,
    pred=None,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> dspy.Prediction:
    query = gold["query"]
    response = gold["answer"]
    score = score_answer(query, response)

    smart_feedback = ""
    new_score = 0
    if score >= 0.95:
        new_score = 1
        smart_feedback += (
            f"Your answer is satisfactory. You should keep doing this good."
        )
    else:
        smart_feedback += (
            f"Sorry, you are answer is not quite satisfactory. You must do better."
        )

    return dspy.Prediction(score=new_score, feedback=smart_feedback)
