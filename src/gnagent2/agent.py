"""
This script reuses the optimized GeneNetwork Agent to address a query
"""

from all_config import program
from query import query

optimized_program = program.load("optimized_program.json")

optimized_program(query=query)
