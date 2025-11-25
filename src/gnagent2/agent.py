"""
This script reuses the optimized GeneNetwork Agent to address a query
"""

import json
import os
import warnings

from gnagent_adapter import GNAgentAdapter

QUERY = os.getenv("QUERY")

with open("optimized_config.json") as f:
    read = f.read()
    optimized_config = json.loads(read)

optimized_agent = GNAgentAdapter(optimized_config)
optimized_agent(query=QUERY)
