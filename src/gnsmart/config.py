"""
This module sets up configurations to run a smarter agentic system based strictly on tool calling
It provides different constructs to interact with the LLM
"""

import json
import os
from pathlib import Path
from typing import Any, Literal

import dspy
import torch
from Bio import Entrez
from Bio.Entrez import efetch, esearch, esummary, read
from langchain_core.messages import BaseMessage
from SPARQLWrapper import JSON, SPARQLWrapper

TTL_PATH = os.getenv("TTL_PATH")
if TTL_PATH is None:
    raise FileNotFoundError(
        "TTL_PATH must be specified to extract RDF schema and build queries"
    )

QUERY = os.getenv("QUERY")
if QUERY is None:
    raise ValueError("QUERY must be specified for program to run")

SEED = os.getenv("SEED")
if SEED is None:
    raise ValueError("SEED must be specified for reproducibility")

EMAIL = os.getenv("EMAIL")
if EMAIL is None:
    raise ValueError("EMAIL must be specified for NCBI tool calling")

Entrez.email = EMAIL


MODEL_NAME = os.getenv("MODEL_NAME")
if MODEL_NAME is None:
    raise ValueError("MODEL_NAME must be specified - either proprietary or local")

MODEL_TYPE = os.getenv("MODEL_TYPE")
if MODEL_TYPE is None:
    raise ValueError("MODEL_TYPE must be specified")

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if int(MODEL_TYPE) == 0:
    GENERATIVE_MODEL = dspy.LM(
        model=f"openai/{MODEL_NAME}",
        api_base="http://localhost:7501/v1",
        api_key="local",
        model_type="chat",
        max_tokens=10_000,
        n_ctx=30_000,
        seed=2_025,
        temperature=0,
        verbose=False,
    )
elif int(MODEL_TYPE) == 1:
    API_KEY = os.getenv("API_KEY")
    if API_KEY is None:
        raise ValueError("Valid API_KEY must be specified to use the proprietary model")
    GENERATIVE_MODEL = dspy.LM(
        MODEL_NAME,
        api_key=API_KEY,
        max_tokens=10_000,
        temperature=0,
        verbose=False,
    )
else:
    raise ValueError("MODEL_TYPE must be 0 or 1")


dspy.configure(lm=GENERATIVE_MODEL)


class Decide(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    next_decision: Literal["researcher", "reflector", "expert", "end"] = (
        dspy.OutputField(desc="The next step to take based on instructions")
    )
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the decision in 50 words"
    )


# Module to manage system
supervise = dspy.Predict(Decide)


class Plan(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The plan to solve the task")
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the output in 50 words"
    )


# Module to make plan
plan = dspy.Predict(Plan)


# Specialized ReAct architecture for researcher


def extract_schema(ttl_path: str) -> tuple[list, list]:
    ttl_files = [
        os.path.join(ttl_path, ttl) for ttl in os.listdir(ttl_path) if "ttl" in ttl
    ]
    prefixes = []
    predicates = []
    for ttl in ttl_files:
        with open(ttl) as f:
            contents = f.readlines()
            for content in contents:
                content = content.strip()
                if content.startswith("@"):
                    content_list = content.split()
                    if len(content_list) == 3:
                        prefix = content_list[1]
                        if prefix not in prefixes:
                            prefixes.append(prefix)
                elif len(content) != 0:
                    content_list = content.split()
                    if len(content_list) == 3:
                        predicate = content_list[1]
                        if predicate not in predicates:
                            predicates.append(predicate)
    return prefixes, predicates


class QueryTranslation(dspy.Signature):
    original_query: str = dspy.InputField()
    rdf_prefixes: list[str] = dspy.InputField()
    triple_predicates: list[str] = dspy.InputField()
    translated_query: str = dspy.OutputField(
        desc="SPARQL query corresponding to user query for fetching requested data given RDF schema inferred from RDF prefixes and triple predicates"
    )


translate_query = dspy.Predict(QueryTranslation)


def fetch_data(query: str) -> Any:
    sparql = SPARQLWrapper("http://sparql-test.genenetwork.org/sparql/")
    sparql.setReturnFormat(JSON)
    if Path(f"{TTL_PATH}/schema.txt").exists():
        with open(f"{TTL_PATH}/schema.txt") as f:
            prefixes, predicates = json.loads(f.read())
    else:
        prefixes, predicates = extract_schema(TTL_PATH)
        with open(f"{TTL_PATH}/schema.txt", "w") as f:
            f.write(json.dumps([prefixes, predicates]))
    sparql_query = translate_query(
        original_query=query, rdf_prefixes=prefixes, triple_predicates=predicates
    ).get("translated_query")
    sparql.setQuery(sparql_query)
    return sparql.queryAndConvert()


fetch_data = dspy.Tool(
    name="fetch_data",
    desc="Fetch RDF data around GeneNetwork data through SPARQL",
    args={
        "query": {
            "type": "string",
            "desc": "SPARQL query to run to fetch relevant data",
        },
    },
    func=fetch_data,
)


class ReactSig(dspy.Signature):
    query: list[BaseMessage] = dspy.InputField()
    solution: str = dspy.OutputField(desc="The answer to the query")


class ReactResearcher(dspy.Module):
    def __init__(self):
        super().__init__()
        self.tools = [fetch_data]

        self.react = dspy.ReAct(
            signature=ReactSig,
            tools=self.tools,
            max_iters=50,  # maximum number of steps for reasoning and tool calling
        )

    def forward(self, query: str):
        return self.react(query=query)


# Specialized ReAct architecture for expert


def search_ncbi(database: str, term: str, max_results: int = 10) -> Any:
    handle = esearch(db=database, term=term, retmax=max_results)
    records = read(handle)
    handle.close()
    return records


search_ncbi = dspy.Tool(
    name="search_ncbi",
    desc="Search an NCBI database (e.g., nucleotide, protein, pubmed) for a term",
    args={
        "database": {
            "type": "string",
            "desc": "Database name like 'nucleotide' or 'pubmed'",
        },
        "term": {"type": "string", "desc": "Search term or query"},
        "max_results": {
            "type": "integer",
            "desc": "Max results (default 10)",
            "default": 10,
        },
    },
    func=search_ncbi,
)


def fetch_record(database: str, record_id: str, rettype: str) -> str:
    handle = efetch(db=database, id=record_id, rettype=rettype, retmode="text")
    result = handle.readline().strip()
    handle.close()
    return result


fetch_record = dspy.Tool(
    name="fetch_record",
    desc="Fetch a record from an NCBI database (e.g., nucleotide, protein, pubmed)",
    args={
        "database": {
            "type": "string",
            "desc": "Database name like 'nucleotide' or 'pubmed'",
        },
        "record_id": {"type": "string", "desc": "Identifier of record"},
        "rettype": {"type": "string", "desc": "Return type compatible with database"},
    },
    func=fetch_record,
)


def summarize_record(database: str, record_id: str) -> Any:
    handle = esummary(db=database, id=record_id)
    result = read(handle)
    handle.close()
    return result


summarize_record = dspy.Tool(
    name="summarize_record",
    desc="Get summary on a record from an NCBI database (e.g., nucleotide, protein, pubmed)",
    args={
        "database": {
            "type": "string",
            "desc": "Database name like 'nucleotide' or 'pubmed'",
        },
        "record_id": {"type": "string", "desc": "Identifier of record"},
    },
    func=summarize_record,
)


class ReactExpert(dspy.Module):
    def __init__(self):
        super().__init__()
        self.tools = [search_ncbi, fetch_record, summarize_record]

        self.react = dspy.ReAct(
            signature=ReactSig,
            tools=self.tools,
            max_iters=50,
        )

    def forward(self, query: list[BaseMessage]):
        return self.react(query=query)


class Tune(dspy.Signature):
    background: list[BaseMessage] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The new questions")
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of the output in 50 words"
    )


# Module to tune reflection
tune = dspy.Predict(Tune)


class End(dspy.Signature):
    messages: list[BaseMessage] = dspy.InputField()
    feedback: str = dspy.OutputField(
        desc="Detailed and comprehensive final feedback combining AI outputs in the list of messages and linking them when necessary"
    )


# Module to wrap up
end = dspy.Predict(End)
