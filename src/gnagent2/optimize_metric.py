"""
This script optimizes the metric used to evaluate GeneNetwork Agent using GEPA
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dspy import GEPA

from all_config import *

train_set, val_set, test_set = get_dataset()


@dataclass
class MetricProgram(dspy.Module):
    """
    Transforms a metric to a dspy program
    Metric should not give feedback
    """
    checker: Callable = match_checker
    judge: dspy.Predict = judge

    def __post_init__(self):
        super().__init__()
    
    def forward(self, example: dspy.Example, prediction: dspyPredidction) -> dspy.Prediction:
        score = self.checker(
            example=example,
            prediction=prediction,
        )
        return dspy.Prediction(score=score)

        
@dataclass
class MetricOptimization
    """
    Wraps optimization of the metric used for evaluation
    """
    program: Any
    reflection_model: Any
    train_set: list[dspy.Example]
    val_set: list[dspy.Example]
    output_path: str = "optimized_metric.json"

    def gepa_optimize(self) -> Any:
        new_metric = 
        optimizer = GEPA(
            metric=new_metric,
            auto="light",
            num_threads=1,
            track_stats=True,
            reflection_lm=self.reflection_model,
            seed=2025,
        )
        optimized_metric = optimizer.compile(
            self.metric,
            trainset=self.train_set,
            valset=self.val_set,
        )
        evaluate(optimized_metric)

        # Save new configurations
        optimized_metric.save(self.output_path)

        return optimized_metric

    
if not Path("optimized_metric.json").exists():
    metric_program = MetricProgram()
    metric_run = MetricOptimization(reflection_model=REFLECTION_MODEL, program=metric_program, train_set=train_set, val_set=val_set)
    match_checker = metric_run.gepa_optimize()
else:
    match_checker = match_checker.load("optimized_metric.json")

# Reuse configurations of match_checker
match_checker_feedback = match_checker_feedback.load("optimized_metric.json")

