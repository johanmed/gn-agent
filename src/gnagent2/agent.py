"""
This script reuses the optimized GeneNetwork Agent to address a query
"""

import json

from gnagent_adapter import GNAgentAdapter
from query import query

with open("optimized_config.json") as f:
    read = f.read()
    optimized_config = json.loads(read)

optimized_agent = GNAgentAdapter(optimized_config)
optimized_agent(query=query)
