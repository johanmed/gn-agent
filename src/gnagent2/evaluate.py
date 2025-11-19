"""
This script compares performance of GeneNetwork Agent before and after prompt optimization
"""

import json

from all_config import *
from gnagent_adapter import GNAgentAdapter, config, extract_config
from optimize_program import get_dataset

train_set, val_set, test_set = get_dataset(
    "examples/general.csv", column_names=["query", "answer", "reasoning"]
)

evaluate = dspy.Evaluate(
    devset=test_set,
    metric=match_checker,
    num_threads=1,
    display_table=True,
    display_progress=True,
    lm=REFLECTION_MODEL,
)


original_agent = GNAgentAdapter(config)
original_result = evaluate(original_agent)

with open("optimized_config.json") as f:
    read = f.read()
    optimized_config = json.loads(read)

optimized_agent = GNAgent(
    corpus_path=optimized_config["corpus_path"],
    pcorpus_path=optimized_config["pcorpus_path"],
    db_path=optimized_config["db_path"],
    naturalize_prompt=optimized_config["prompts"]["naturalize_prompt"],
    rephrase_prompt=optimized_config["prompts"]["rephrase_prompt"],
    analyze_prompt=optimized_config["prompts"]["analyze_prompt"],
    check_prompt=optimized_config["prompts"]["check_prompt"],
    summarize_prompt=optimized_config["prompts"]["summarize_prompt"],
    synthesize_prompt=optimized_config["prompts"]["synthesize_prompt"],
    split_prompt=optimized_config["prompts"]["split_prompt"],
    finalize_prompt=optimized_config["prompts"]["finalize_prompt"],
    sup_prompt1=optimized_config["prompts"]["sup_prompt1"],
    sup_prompt2=optimized_config["prompts"]["sup_prompt2"],
    plan_prompt=optimized_config["prompts"]["plan_prompt"],
    refl_prompt=optimized_config["prompts"]["refl_prompt"],
)
final_config = extract_config(optimized_agent)
final_agent = GNAgentAdapter(final_config)

optimized_result = evaluate(final_agent)
