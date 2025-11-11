"""
This script optimizes prompts of GeneNetwork Agent using GEPA
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dspy import GEPA

from all_config import *

dspy.settings.configure(constrain_outputs=False)

train_set, val_set, test_set = get_dataset()

evaluate = dspy.Evaluate(
    devset=test_set,
    metric=match_checker,
    num_threads=1,
    display_table=True,
    display_progress=True,
)

evaluate(program)

@dataclass
class GepaOptimization:
    """
    Wraps GEPA optimization of GeneNetwork Agent's prompts using a reflection model
    """
    gnagent_program: Any
    reflection_model: Any
    prompts: list[str] = field(default_factory=lambda: [
        "plan",
        "tune",
        "supervise",
        "end",
        "rephrase",
        "analyze",
        "check",
        "summarize",
        "synthesize",
        "subquery",
        "finalize",
    ])
    output_path: str = "optimized_program.json"
    
    def gepa_optimize(self) -> Any:
        
        optimizer = GEPA(
            metric=match_checker_feedback,
            auto="light",
            num_threads=1,
            track_stats=True,
            reflection_lm=self.reflection_model,
            seed=2025,
        )

        optimized_program = optimizer.compile(
            program,
            trainset=train_set,
            valset=val_set,
        )

        evaluate(optimized_program)

        # Save new configurations
        optimized_program.save(self.output_path)

        return optimized_program

run = GepaOptimization(gnagent_program=program, reflection_model=REFLECTION_MODEL)
run.gepa_optimize()
