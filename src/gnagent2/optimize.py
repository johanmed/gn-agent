"""
This scripts optimizes GeneNetwork Agent using GEPA
"""

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
    lm=REFLECTION_MODEL,
)

evaluate(program)

optimizer = GEPA(
    metric=match_checker_feedback,
    auto="light",
    num_threads=1,
    track_stats=True,
    reflection_minibatch_size=3,
    reflection_lm=REFLECTION_MODEL,
)

optimized_program = optimizer.compile(
    program,
    trainset=train_set,
    valset=val_set,
)

evaluate(optimized_program)

optimized_program.save("optimized_program.json")

