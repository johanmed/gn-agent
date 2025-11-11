"""
This script optimizes prompts of GeneNetwork Agent using GEPA
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dspy import GEPA

from all_config import *


train_set, val_set, test_set = get_dataset()

evaluate = dspy.Evaluate(
    devset=test_set,
    metric=match_checker,
    num_threads=1,
    display_table=True,
    display_progress=True,
    lm=REFLECTION_MODEL,
)

evaluate(program)

@dataclass
class ProgramOptimization:
    """
    Wraps GEPA optimization of GeneNetwork Agent's prompts using a reflection model
    """
    program: Any
    reflection_model: Any
    metric : Any
    train_set: list[dspy.Example]
    val_set: list[dspy.Example]
    output_path: str = "optimized_program.json"
    
    def gepa_optimize(self) -> Any:
        optimizer = GEPA(
            metric=self.metric,
            auto="light",
            num_threads=1,
            track_stats=True,
            reflection_lm=self.reflection_model,
            seed=2025,
        )
        optimized_program = optimizer.compile(
            self.program,
            trainset=self.train_set,
            valset=self.val_set,
        )
        evaluate(optimized_program)

        # Save new configurations
        optimized_program.save(self.output_path)

        return optimized_program

if not Path("optimized_program.json").exists():
    program_run = ProgramOptimization(program=program, reflection_model=REFLECTION_MODEL, metric=match_checker_feedback, train_set=train_set, val_set=val_set)
    program_run.gepa_optimize()
    
