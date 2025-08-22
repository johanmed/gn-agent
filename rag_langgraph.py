"""
This scripts runs through a demo on the use of RAG system for genomic analysis
Reasoning based on tree-of-thought
Embedding model = Qwen/Qwen3-Embedding-0.6B
Generative model = calme-3.2-instruct-78b-Q4_K_S
Summary model = Phi-3-mini-4k-instruct-fp16
"""
import click
import os
import sys
import time

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from typing_extensions import TypedDict

from transformers import AutoTokenizer
from langchain.memory import ConversationSummaryBufferMemory
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma

from langgraph.graph import StateGraph, START
import asyncio

from rdflib import Graph
from glob import glob

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# XXX: Remove hard-coded path.
CORPUS_PATH="/home/johannesm/corpus/"

# XXX: Remove hard-coded path.
DB_PATH="/home/johannesm/tmp/chroma_db"

EMBED_MODEL="Qwen/Qwen3-Embedding-0.6B"

# XXX: Remove hard-coded paths.
GENERATIVE_MODEL=LlamaCpp(
    model_path="/home/johannesm/pretrained_models/calme-3.2-instruct-78b-Q4_K_S.gguf",
    max_tokens=200,
    n_ctx=32_768,
    seed=2_025,
    temperature=0,
    verbose=False)

# XXX: Remove hard-coded paths.
SUMMARY_MODEL=LlamaCpp(
    model_path="/home/johannesm/pretrained_models/Phi-3-mini-4k-instruct-fp16.gguf",
    max_tokens=100,
    n_ctx=4096,
    seed=2025,
    temperature=0,
    verbose=False)

class State(TypedDict):
    input: str
    chat_history: list
    context: list
    answer: str
    
@dataclass
class GNQNA_RAG():
    corpus_path: str
    db_path: str
    chroma_db: Any = field(init=False)
    docs: list = field(init=False)
    ensemble_retriever: Any = field(init=False)
    
    def __post_init__(self):
        self.docs=self.corpus_to_docs(self.corpus_path)
        self.chroma_db=self.set_chroma_db(
            docs=self.docs,
            embed_model=HuggingFaceEmbeddings(
                model_name=EMBED_MODEL),
            db_path=self.db_path)

        # Init'ing the ensemble retriever
        bm25_retriever=BM25Retriever.from_texts(self.docs)
        bm25_retriever.k=5   # KLUDGE: Explain why the magic number 5
        self.ensemble_retriever=EnsembleRetriever(
            retrievers=[self.chroma_db.as_retriever(), bm25_retriever],
            weights=[0.3, 0.7])  # KLUDGE: Explain why the magic array

    
    def corpus_to_docs(self, corpus_path: str) -> list:
        """Convert a corpus into an array of sentences.
        KLUDGE: XXXX: Corpus of text should be RDF.  This here
        is for testing.
        """
        start=time.time()
        # Check for corpus. Exit if no corpus.
        if not Path(corpus_path).exists():
            sys.exit(1)
        turtles=glob(f"{corpus_path}rdf_data*.ttl")
        g=Graph()
        for turtle in turtles:    
            g.parse(turtle, format='turtle')
        docs=[]
        for subject in set(g.subjects()):
            text=f"Entity {subject}\n"
            for predicate, obj in g.predicate_objects(subject):
                text+=f"has {predicate} of {obj}\n"
            docs.append(text)
        end=time.time()
        print(f'corpus_to_docs: {end-start}')
        return docs
    
    def set_chroma_db(self, docs: list,
                      embed_model: Any, db_path: str,
                      chunk_size: int = 500) -> Any:
        match Path(db_path).exists():
            case True:
                db=Chroma(
                    persist_directory=db_path,
                    embedding_function=embed_model
                )
                return db
            case _:
                for i in tqdm(range(0, len(docs), chunk_size)):
                    chunk=docs[i:i+chunk_size]
                    db=Chroma.from_texts(
                        texts=chunk,
                        embedding=embed_model,
                        persist_directory=db_path)
                    db.persist()
                return db

    def retrieve(self, state: State) -> Any:
        # Define graph node for retrieval
        prompt=f"""
        You are powerful data retriever and you strictly return
        what is asked for.
        Retrieve relevant documents for the query below:
        {state['input']}
        """
        tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL)
        tokenized_prompt=tokenizer(prompt, padding=True)
        retrieved_docs=self.ensemble_retriever.invoke(tokenized_prompt)
        return {"context": retrieved_docs}

    def generate(self, state:State) -> Any:
        # Define graph node for generation
        context="\n".join(info.page_content for info in state["context"])
        prompt= f"""
             <|im_start|>system
             You are a very skilled analyst that can answer concisely
             questions based on information you have.
             <|im_end|>
             <|im_start|>user
             Answer the question below using following information.
             Try using the history first.
             If you cannot provide answer with it, then use the context
             Context: {context}
             History: {state["chat_history"]}
             Question: {state["input"]}
             Answer:
             <|im_end|>
             <|im_start|>assistant"""
        response = GENERATIVE_MODEL.invoke(prompt)
        return {"answer": response}


    def summarize(self, state:State) -> Any:
        # Define node for history summarization
        return ConversationSummaryBufferMemory(
            llm=SUMMARY_MODEL,
            memory_key="chat_history",
            max_token_limit=200,
            prompt=f"""
            <|system|>
            You are an excellent and concise summary maker.
            <|end|>
            <|user|>
            Please provide a short summary of the information above.
            state["chat_history"]
            state["input"]
            state["answer"]
            <|end|>
            <|assistant|>
            """,
            return_messages=True)
        
    def initialize_langgraph_chain(self) -> Any:
        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("generate", self.generate)
        graph_builder.add_node("summarize", self.summarize)
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", "summarize")
        graph = graph_builder.compile()
        return graph

    async def invoke_langgraph(self, question: str) -> Any:
        graph = self.initialize_langgraph_chain()
        result = await graph.ainvoke(
                {"input": question}) # Run graph asynchronously
        return result["answer"]

    
    def retrieve_generate(self, question: str) -> Any:
        start=time.time()
        result=asyncio.run(self.invoke_langgraph(question))
        # Close LLMs
        GENERATIVE_MODEL.client.close()
        SUMMARY_MODEL.client.close()
        end=time.time()
        print(f'retrieve_generate: {end-start}')
        return {"result": result}

rag=GNQNA_RAG(
    corpus_path=CORPUS_PATH,
    db_path=DB_PATH)
#query=input('Please enter your query:')
output=rag.retrieve_generate('I want you to extract from the database anywhere there is D12mit280. You allowed to initiate many rounds of retrieval until you get 20 relevant results. Next, extract the lod score and trait for each result. List for me traits that have a lod score > 4.0. Join to the list the corresponding lod scores so I can confirm. Show results using the following format: trait - lod score\n')
print(output['result'])
