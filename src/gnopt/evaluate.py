"""This module compares performance of GeneNetwork Agent on a specific task before and after user prompt optimization"""

import logging

import dspy

from adapter import dspy_agent
from optimize import get_dataset

logging.basicConfig(
    filename="log_evaluation.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


def assess_performance(example_path: str) -> dspy.EvaluationResult:
    train_set, val_set, test_set = get_dataset(
        example_path, column_names=["query", "answer", "reasoning"]
    )
    evaluate = dspy.Evaluate(
        devset=test_set,
        metric=assesser,
        num_threads=1,
        display_table=False,
        display_progress=True,
        lm=GENERATIVE_MODEL,
    )
    return evaluate(dspy_agent).get("results")


task = input("Genomic task to perform: ")
original = assess_performance(f"examples/{task}/original.csv")
optimized = assess_performance(f"examples/{task}/optimized.csv")

logging.info(
    f"The performance before optimization:\n{original}\n\nThe performance after optimization:\n{optimized}"
)
