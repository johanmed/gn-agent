"""
This script optimizes prompts of GeneNetwork Agent using GEPA
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from dspy import GEPA

from all_config import *
from gnagent_adapter import GNAgentAdapter, config


def get_dataset(
    example_path: str,
    split_ratio: int = 0.7,
    column_names: list[str] = ["query", "prompt_output", "prompt_text", "reasoning"],
) -> Any:
    data = pd.read_csv(
        example_path, names=column_names, nrows=25
    )  # need to remove limit
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
    Wraps GEPA optimization of any dspy module using a reflection model
    """

    module: Any
    metric: Any
    input_path: str
    train_set: list[dspy.Example] = field(init=False)
    val_set: list[dspy.Example] = field(init=False)

    def __post_init__(self):
        train_set, val_set, test_set = get_dataset(self.input_path)
        val_set = val_set + test_set
        self.train_set = train_set
        self.val_set = val_set

    def optimize(self) -> Any:
        optimizer = GEPA(
            metric=self.metric,
            auto="light",
            num_threads=6,
            track_stats=True,
            reflection_lm=REFLECTION_MODEL,
            seed=2025,
        )
        optimized_module = optimizer.compile(
            self.module,
            trainset=self.train_set,
            valset=self.val_set,
        )

        return optimized_module


if __name__ == "__main__":
    if not Path("optimized_config.json").exists():
        adapter = GNAgentAdapter(config)
        optimized_prompts: Dict[str, str] = {}

        for name, predictor in adapter.named_predictors():
            REFLECTION_MODEL.history = []
            original = copy.deepcopy(predictor.signature)
            pred = dspy.Predict(original)

            input_path = f"examples/{name}.csv"
            if Path(input_path).exists():
                logging.info(f"Proceeding to optimization for {name}")
                module_run = Optimization(
                    module=pred,
                    metric=match_checker_feedback,
                    input_path=input_path,
                )
                optimized_predictor = module_run.optimize()
                best_prompt = extract_best_prompt(REFLECTION_MODEL)
                optimized_prompts[name] = best_prompt.strip()
            else:
                logging.warning(f"No examples for {name}???")

        new_config = copy.deepcopy(config)
        new_config["prompts"].update(optimized_prompts)
        with open("optimized_config.json", "w") as f:
            f.write(json.dumps(new_config))
        logging.info("Optimization complete and prompts saved!")
    else:
        logging.warning("GNAgent already optimized!")
