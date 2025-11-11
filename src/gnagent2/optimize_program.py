"""
This script optimizes prompts of GeneNetwork Agent using GEPA
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dspy import GEPA

from all_config import *

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

        # Predictors for prompts to improve
        self.plan = dspy.ChainOfThought(
            type("PlanSig", (Plan,), {"__doc__": self.gn_agent.plan_prompt.content})
        )
        self.tune = dspy.ChainOfThought(
            type("TuneSig", (Tune,), {"__doc__": self.gn_agent.refl_prompt.content})
        )
        self.sup1 = dspy.ChainOfThought(
            type("Sup1Sig", (Decide,), {"__doc__": self.gn_agent.sup_prompt1.content})
        )
        self.sup2 = dspy.ChainOfThought(
            type("Sup2Sig", (Decide,), {"__doc__": self.gn_agent.sup_prompt2.content})
        )
        self.end = dspy.Predict(
            type("EndSig", (End,), {"__doc__": self.gn_agent.finalize_prompt.content})
        )
        self.naturalize = dspy.Predict(
            type(
                "NaturalizeSig",
                (Naturalize,),
                {"__doc__": self.gn_agent.naturalize_prompt.content},
            )
        )
        self.rephrase = dspy.Predict(
            type(
                "RephraseSig",
                (Rephrase,),
                {"__doc__": self.gn_agent.rephrase_prompt.content},
            )
        )
        self.analyze = dspy.ChainOfThought(
            type(
                "AnalyzeSig",
                (Analyze,),
                {"__doc__": self.gn_agent.analyze_prompt.content},
            )
        )
        self.check = dspy.Predict(
            type("CheckSig", (Check,), {"__doc__": self.gn_agent.check_prompt.content})
        )
        self.summarize = dspy.Predict(
            type(
                "SummarizeSig",
                (Summarize,),
                {"__doc__": self.gn_agent.summarize_prompt.content},
            )
        )
        self.synthesize = dspy.ChainOfThought(
            type(
                "SynthesizeSig",
                (Synthesize,),
                {"__doc__": self.gn_agent.synthesize_prompt.content},
            )
        )
        self.subquery = dspy.Predict(
            type(
                "SubquerySig",
                (Subquery,),
                {"__doc__": self.gn_agent.split_prompt.content},
            )
        )
        self.finalize = dspy.Predict(
            type(
                "FinalizeSig",
                (Finalize,),
                {"__doc__": self.gn_agent.finalize_prompt.content},
            )
        )

    def run_handler(self, query):
        # Runs async handler in clean event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.gn_agent.handler(query))
        finally:
            loop.close()

    def forward(self, query):
        # Runs async call in thread
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
    metric: Any
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
    program_run = ProgramOptimization(
        program=program,
        reflection_model=REFLECTION_MODEL,
        metric=match_checker_feedback,
        train_set=train_set,
        val_set=val_set,
    )
    program_run.gepa_optimize()
