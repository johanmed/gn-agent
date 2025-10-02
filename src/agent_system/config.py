"""
Setup and LLMs

1. Embedding model = Qwen/Qwen3-Embedding-0.6B
2. Generative model = calme-3.2-instruct-78b-Q4_K_S (very large model)
3. Summary model = Phi-3-mini-4k-instruct (small model)
"""

import logging
import warnings

from langchain_community.llms import LlamaCpp

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(
    filename="log_langgraph.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


# XXX: Remove hard-coded path.
CORPUS_PATH = "/home/johannesm/rdf_corpus/"

# XXX: Remove hard_coded path.
PCORPUS_PATH = "/home/johannesm/rdf_tmp/docs.txt"

# XXX: Remove hard-coded path.
DB_PATH = "/home/johannesm/rdf_tmp/chroma_db"

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# XXX: Remove hard-coded paths.
GENERATIVE_MODEL = LlamaCpp(
    model_path="/home/johannesm/pretrained_models/calme-3.2-instruct-78b-Q4_K_S.gguf",
    max_tokens=10_000,
    n_ctx=32_768,
    seed=2_025,
    temperature=0,
    verbose=False,
)

# XXX: Remove hard-coded paths.
SUMMARY_MODEL = LlamaCpp(
    model_path="/home/johannesm/pretrained_models/Phi-3-mini-4k-instruct-fp16.gguf",
    max_tokens=1_000,
    n_ctx=4_096,
    seed=2025,
    temperature=0,
    verbose=False,
)
