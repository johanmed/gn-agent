"""
This script compares performance of GeneNetwork Agent before and after prompt optimization
"""

import json

from all_config import *
from gnagent_adapter import GNAgentAdapter
from optimize_program import get_dataset

train_set, val_set, test_set = get_dataset(
    "examples/general.csv", column_names=["query", "output", "reasoning"]
)

evaluate = dspy.Evaluate(
    devset=test_set,
    metric=match_checker,
    num_threads=1,
    display_table=True,
    display_progress=True,
)

original_agent = agent = GNAgent(
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

evaluate(original_agent)

with open("optimized_config.json") as f:
    read = f.read()
    optimized_config = json.loads(read)

optimized_agent = GNAgentAdapter(optimized_config)

evaluate(optimized_agent)
