"""
This script tests the GeneNetwork Agent system
"""

import os
import sys

from src.agent.agent import GNAgent


def test_agent():
    agent = GNAgent(
        corpus_path="CORPUS_PATH",
        pcorpus_path="PCORPUS_PATH",
        db_path="DB_PATH",
        naturalize_prompt="naturalize_prompt",
        analyze_prompt="analyze_prompt",
        check_prompt="check_prompt",
        summarize_prompt="summarize_prompt",
        synthesize_prompt="synthesize_prompt",
        split_prompt="split_prompt",
        finalize_prompt="finalize_prompt",
        sup_system_prompt1="sup_system_prompt1",
        sup_system_prompt2="sup_system_prompt2",
        plan_system_prompt="plan_system_prompt",
        refl_system_prompt="refl_system_prompt",
    )
    assert (
        agent.corpus_path == "CORPUS_PATH"
        and agent.refl_system_prompt == "refl_system_prompt"
    )
