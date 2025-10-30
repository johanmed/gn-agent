"""
This script tests query passed to system
"""

import os
import sys

from src.agent.query import make_query_prompt


def test_query():
    question = "What is a gene?"
    query = make_query_prompt(query=question)
    assert (
        isinstance(query, str)
        and query
        == "Use all resources at your disposal to answer the following question: What is a gene?"
    )
