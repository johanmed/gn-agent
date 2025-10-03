"""
Setup and LLMs

1. Embedding model = Qwen/Qwen3-Embedding-0.6B
2. Generative model = calme-3.2-instruct-78b-Q4_K_S (very large model)
3. Summary model = Phi-3-mini-4k-instruct (small model)
"""

import logging
import warnings

import dspy

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(
    filename="log_langgraph.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


# X: Remove hard-coded path.
CORPUS_PATH = "/home/johannesm/rdf_corpus/"

# X: Remove hard_coded path.
PCORPUS_PATH = "/home/johannesm/rdf_tmp/docs.txt"

# X: Remove hard-coded path.
DB_PATH = "/home/johannesm/rdf_tmp/chroma_db"


EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

GENERATIVE_MODEL = dspy.LM(
    model="openai/MaziyarPanahi/calme-3.2-instruct-78b",
    api_base="http://localhost:7501/v1",
    api_key="local",
    model_type="chat",
    max_tokens=10_000,
    n_ctx=32_768,
    seed=2_025,
    temperature=0,
    verbose=False,
)

SUMMARY_MODEL = dspy.LM(
    model="microsoft/Phi-3-mini-4k-instruct",
    api_base="http://localhost:7502/v1",
    api_key="local",
    model_type="chat",
    max_tokens=1_000,
    n_ctx=4_096,
    seed=2025,
    temperature=0,
    verbose=False,
)

deep_generate = dspy.ChainOfThought("question -> answer: str", lm=GENERATIVE_MODEL)

shallow_generate = dspy.ChainOfThought("question -> answer: str", lm=SUMMARY_MODEL)
