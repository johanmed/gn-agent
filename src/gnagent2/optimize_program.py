"""
This script optimizes prompts of GeneNetwork Agent using GEPA
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from dspy import GEPA
from langchain_core.messages import SystemMessage

from all_config import *
from gnagent_adapter import adapter


train_set, val_set, test_set = get_dataset()

agent = GNAgent(
    corpus_path=CORPUS_PATH,
    pcorpus_path=PCORPUS_PATH,
    db_path=DB_PATH,
    naturalize_prompt=naturalize_prompt,
    rephrase_prompt=rephrase_prompt,
    analyze_prompt=analyze_prompt,
    check_prompt=check_prompt,
    summarize_prompt=summarize_prompt,
    synthesize_prompt=synthesize_prompt,
    split_prompt=split_prompt,
    finalize_prompt=finalize_prompt,
    sup_prompt1=sup_prompt1,
    sup_prompt2=sup_prompt2,
    plan_prompt=plan_prompt,
    refl_prompt=refl_prompt,
)


class GNAgentProgram(dspy.Module):
    """
    Transforms GNAgent to a dspy Program
    """
    
    def __init__(self, gn_agent: GNAgent):
        super().__init__()
        self.gn_agent = gn_agent
        self.executor = ThreadPoolExecutor(max_workers=1)
            
    def run_handler(self, query):
        # Runs async handler in clean event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.gn_agent.handler(query))
        finally:
            loop.close()

    def forward(self, query) -> dspy.Prediction:
        answer, reasoning = self.executor.submit(self.run_handler, query).result()
        return dspy.Prediction(
            answer=str(answer).strip(), reasoning=str(reasoning).strip()
        )


program = GNAgentProgram(agent)


evaluate = dspy.Evaluate(
    devset=test_set,
    metric=match_checker,
    num_threads=1,
    display_table=True,
    display_progress=True,
)

evaluate(program)


@dataclass
class ProgramOptimization:
    """
    Wraps GEPA optimization of GeneNetwork Agent's prompts using a reflection model
    """

    program: Any  # Must be GNAgent adapter
    reflection_model: Any
    metric: Any
    train_set: list[dspy.Example]
    val_set: list[dspy.Example]
    output_path: str = "optimized_program.json"

    def gepa_optimize(self) -> Any:
        optimizer = GEPA(
            metric=self.metric,
            max_metric_calls=5,
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
        optimized_program.save(self.output_path)

        return optimized_program


if __name__ == "__main__":
    if not Path("optimized_program.json").exists():
        program_run = ProgramOptimization(
            program=adapter,
            reflection_model=REFLECTION_MODEL,
            metric=match_checker_feedback,
            train_set=train_set,
            val_set=val_set,
        )
        program_run.gepa_optimize()
    else:
        print("optimized_program.json already exists!")
        optimized_program = program.load("optimized_program.json")
        evaluate(optimized_program)
