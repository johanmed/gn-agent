"""
This script tests the system configurations
"""

import os
import sys

module_dir = os.path.abspath("../src/agent/")
sys.path.append(module_dir)

from config import generate


def test_generate():
    question = "What is a gene?"
    answer = generate(question=question)
    answer = answer.get("answer")
    assert isinstance(answer, str) and "gene" in answer.lower()
