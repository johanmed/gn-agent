"""This module optimizes user prompt of a specific task using GEPA"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dspy
import pandas as pd
from dspy import GEPA

from adapter import dspy_agent
from config import *

logging.basicConfig(
    filename="log_optimization.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


def get_dataset(
    example_path: str,
    split_ratio: int = 0.7,
    column_names: list[str] = ["query", "answer"],
) -> Any:
    data = pd.read_csv(
        example_path,
        names=column_names,
    )
    data_dicts = data[column_names].to_dict(orient="records")

    formatted = [
        dspy.Example({name: x[name] for name in column_names}).with_inputs(
            column_names[0]
        )
        for x in data_dicts
    ]

    random.Random(2025).shuffle(formatted)
    train_set = formatted[: int(split_ratio * len(formatted))]
    eval_set = formatted[int(split_ratio * len(formatted)) :]

    # Always use 50-50 for validation and test sets
    val_set = eval_set[: int(0.5 * len(eval_set))]
    test_set = eval_set[int(0.5 * len(eval_set)) :]

    return train_set, val_set, test_set


def extract_best_prompt(reflection_lm) -> str:
    """
    Extract best prompt suggested by GEPA
    """
    best_prompt = ""
    for interaction in reflection_lm.history:
        messages = interaction.get("messages")
        if messages is not None:
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role == "system":
                    new_prompt = content.strip()
                    if len(new_prompt) > len(best_prompt):
                        best_prompt = new_prompt
    return best_prompt


@dataclass
class Optimization:
    """
    Wraps GEPA optimization of user prompt using a reflection model
    """

    program: Any
    metric: Any
    example_path: str
    train_set: list[dspy.Example] = field(init=False)
    val_set: list[dspy.Example] = field(init=False)

    def __post_init__(self):
        train_set, val_set, test_set = get_dataset(self.example_path)
        val_set = val_set + test_set
        self.train_set = train_set
        self.val_set = val_set

    def optimize(self) -> Any:
        optimizer = GEPA(
            metric=self.metric,
            max_metric_calls=10,
            num_threads=1,
            track_stats=True,
            reflection_lm=REFLECTION_MODEL,
            seed=2025,
        )
        optimized_program = optimizer.compile(
            self.program,
            trainset=self.train_set,
            valset=self.val_set,
        )

        return optimized_program


if __name__ == "__main__":
    task = input("Genomic task to optimize for: ")
    example_path = f"examples/{task}/original.csv"
    prompt_path = f"prompts/{task}.py"
    if not Path(prompt_path).exists():
        program = Optimization(
            program=dspy_agent,
            metric=smart_assesser,
            example_path=example_path,
        )
        optimized_program = program.optimize()
        best_prompt = extract_best_prompt(REFLECTION_MODEL)
        with open(prompt_path) as f:
            f.write(best_prompt)
        logging.info("Optimization completed and user prompt saved for the task!")
    else:
        logging.warning("User prompt for the task already optimized!")
