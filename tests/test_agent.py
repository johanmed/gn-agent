"""
This script tests the configurations of GeneNetwork Agent
"""

from gnagent.agent import GNAgent


def test_agent():
    agent = GNAgent(
        corpus_path="CORPUS_PATH",
        pcorpus_path="PCORPUS_PATH",
        db_path="DB_PATH",
        naturalize_prompt="naturalize_prompt",
        rephrase_prompt="rephrase_prompt",
        analyze_prompt="analyze_prompt",
        check_prompt="check_prompt",
        summarize_prompt="summarize_prompt",
        synthesize_prompt="synthesize_prompt",
        split_prompt="split_prompt",
        finalize_prompt="finalize_prompt",
        sup_prompt1="sup_prompt1",
        sup_prompt2="sup_prompt2",
        plan_prompt="plan_prompt",
        refl_prompt="refl_prompt",
    )
    assert (
        agent.corpus_path == "CORPUS_PATH"
        and agent.refl_prompt == "refl_prompt"
    )
