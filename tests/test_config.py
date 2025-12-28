"""
This script tests the system configurations
"""

from gnagent.config import extract
from langchain_core.messages import HumanMessage

def test_extract():
    question = "What is a gene?"
    answer = extract(background=[HumanMessage(question)])
    answer = answer.get("answer")
    assert isinstance(answer, str) and "gene" in answer.lower()
