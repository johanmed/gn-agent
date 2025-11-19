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
logging.info(f"Score of original GNAgent: {original_result}")

with open("optimized_config.json") as f:
    read = f.read()
    optimized_config = json.loads(read)

final_agent = GNAgentAdapter(optimized_config)
optimized_result = evaluate(final_agent)
logging.info(f"Score of optimized GNAgent: {optimized_result}")
