"""
This a multi-agent system for genomic analysis

Author: Johannes Medagbe
Copyright 2025
"""

import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from threading import Lock
from typing import Any

from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from rdflib import Graph
from tqdm import tqdm
from typing_extensions import TypedDict

from config import *
from prompts import *
from question import question


class State(TypedDict):
    input: str
    chat_history: list[str]
    context: list[str]
    answer: str
    should_continue: str


@dataclass
class GNQNA:
    corpus_path: str
    pcorpus_path: str
    db_path: str
    chroma_db: Any = field(init=False)
    docs: list = field(init=False)
    ensemble_retriever: Any = field(init=False)
    generative_lock: Lock = field(init=False, default_factory=Lock)
    summary_lock: Lock = field(init=False, default_factory=Lock)
    retriever_lock: Lock = field(init=False, default_factory=Lock)

    def __post_init__(self):

        if not Path(self.pcorpus_path).exists():
            self.docs = self.corpus_to_docs(self.corpus_path)
            with open(self.pcorpus_path, "w") as file:
                file.write(json.dumps(self.docs))
        else:
            with open(self.pcorpus_path) as file:
                data = file.read()
                self.docs = json.loads(data)

        self.chroma_db = self.set_chroma_db(
            docs=self.docs,
            embed_model=HuggingFaceEmbeddings(model_name=EMBED_MODEL),
            db_path=self.db_path,
        )

        # Init'ing the ensemble retriever
        # Explain magic numbers and magic array
        bm25_retriever = BM25Retriever.from_texts(self.docs, k=10)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.chroma_db.as_retriever(search_kwargs={"k": 10}),
                bm25_retriever,
            ],
            weights=[0.4, 0.6],
        )

    def corpus_to_docs(self, corpus_path: str) -> list:
        logging.info("In corpus_to_docs")

        # Check for corpus. Exit if no corpus.
        if not Path(corpus_path).exists():
            sys.exit(1)

        turtles = glob(f"{corpus_path}*.rdf")
        g = Graph()
        for turtle in turtles:
            g.parse(turtle, format="ttl")

        docs = []
        total = len(set(g.subjects()))

        for subject in tqdm(set(g.subjects())):
            text = f"{subject}:"
            for predicate, obj in g.predicate_objects(subject):
                text += f"{predicate}:{obj}\n"

            prompt = naturalize_prompt

            with self.generative_lock:
                response = GENERATIVE_MODEL.invoke(prompt)
            # print(f"Documents: {response}")

            docs.append(response)

            if len(docs) > total / 10:
                break

        return docs

    def set_chroma_db(
        self, docs: list, embed_model: Any, db_path: str, chunk_size: int = 500
    ) -> Any:
        logging.info("In set_chroma_db")
        if Path(db_path).exists():
            db = Chroma(persist_directory=db_path, embedding_function=embed_model)
            return db
        else:
            for i in tqdm(range(0, len(docs), chunk_size)):
                chunk = docs[i : i + chunk_size]
                document = Document(
                    page_content=chunk, metadata={"source": f"Document {i}"}
                )
                db = Chroma.from_texts(
                    texts=document, embedding=embed_model, persist_directory=db_path
                )
                db.persist()
            return db

    def retrieve(self, state: State) -> dict:

        # Retrieve documents
        logging.info("\nRetrieving")

        prompt = retriever_prompt

        with self.generative_lock:
            response = GENERATIVE_MODEL.invoke(prompt)
        logging.info(f"\nResponse in retrieve: {response}")

        if isinstance(response, str):
            start = response.find("[")
            end = response.rfind("]") + 1  # offset by 1 for slicing
            response = json.loads(response[start:end])
        else:
            response = []

        retrieved_docs = []
        with self.retriever_lock:
            for query in response:
                if query:
                    retrieved_docs.append(self.ensemble_retriever.invoke(query))

        new_docs = [
            doc.page_content
            for doc_list in retrieved_docs
            for doc in doc_list
            if hasattr(doc, "page_content")
        ]

        logging.info(f"Retrieved docs in retrieve: {new_docs}")

        should_continue = "analyze"

        return {
            "input": state["input"],
            "context": new_docs,
            "should_continue": should_continue,
            "chat_history": state.get("chat_history", []),
            "answer": state.get("answer", ""),
        }

    def analyze(self, state: State) -> dict:

        # Analyze documents
        logging.info("\nAnalysing")

        context = (
            "\n".join(state.get("context", [])) if state.get("context", []) else ""
        )

        existing_history = (
            "\n".join(state.get("chat_history", []))
            if state.get("chat_history", [])
            else ""
        )

        prompt = analyze_prompt

        with self.generative_lock:
            response = GENERATIVE_MODEL.invoke(prompt)
            response = " ".join(response.split(" ")[:200])  # constraint
        logging.info(f"\nResponse in analyze: {response}")

        should_continue = "check_relevance"

        return {
            "input": state["input"],
            "answer": response,
            "should_continue": should_continue,
            "context": state.get("context", []),
            "chat_history": state.get("chat_history", []),
        }

    def check_relevance(self, state: State) -> dict:

        # Check relevance of retrieved data
        logging.info("\nChecking relevance")

        answer = state["answer"]

        prompt = check_prompt

        with self.summary_lock:
            assessment = SUMMARY_MODEL.invoke(prompt)
        logging.info(f"\nAssessment in checking relevance: {assessment}")

        if "yes" in assessment.lower():
            should_continue = "summarize"
        else:
            should_continue = "end"
            answer = "Sorry, we are unable to \
                provide a valuable feedback due to lack of relevant data."

        return {
            "input": state["input"],
            "context": state.get("context", []),
            "answer": answer,
            "chat_history": state.get("chat_history", []),
            "should_continue": should_continue,
        }

    def summarize(self, state: State) -> dict:

        # Summarize
        logging.info("\nSummarizing")

        existing_history = state.get("chat_history", [])

        current_interaction = f"""
            User: {state["input"]}\nAssistant: {state["answer"]}"""

        full_context = (
            "\n".join(existing_history) + "\n" + current_interaction
            if existing_history
            else current_interaction
        )

        prompt = summarize_prompt

        with self.summary_lock:
            summary = SUMMARY_MODEL.invoke(prompt)

        if not summary or not isinstance(summary, str) or summary.strip() == "":
            summary = f"- {state['input']} - No valid answer generated"

        updated_history = existing_history + [summary]  # update chat_history
        logging.info(f"\nChat history in summarize: {updated_history}")

        # Generate final answer
        if not updated_history:
            final_answer = "Insufficient data for analysis."
        else:
            prompt = synthesize_prompt

            with self.generative_lock:
                response = GENERATIVE_MODEL.invoke(prompt)
            logging.info(f"Answer in summarize: {response}")

            proc_answer = (
                response
                if response
                else "Sorry, we are unable to \
            provide a valuable feedback due to lack of relevant data."
            )

        return {
            "input": state["input"],
            "answer": proc_answer,
            "context": state.get("context", []),
            "chat_history": updated_history,
        }

    def initialize_langgraph_chain(self) -> Any:
        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("check_relevance", self.check_relevance)
        graph_builder.add_node("analyze", self.analyze)
        graph_builder.add_node("summarize", self.summarize)
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "analyze")
        graph_builder.add_edge("analyze", "check_relevance")
        graph_builder.add_edge("summarize", END)
        graph_builder.add_conditional_edges(
            "check_relevance",
            lambda state: state.get("should_continue", "summarize"),
            {"summarize": "summarize", "end": END},
        )
        graph = graph_builder.compile()

        return graph

    async def invoke_langgraph(self, question: str) -> Any:
        graph = self.initialize_langgraph_chain()
        initial_state = {
            "input": question,
            "chat_history": [],
            "context": [],
            "answer": "",
            "should_continue": "retrieve",
        }

        result = await graph.ainvoke(initial_state)

        return result

    def split_query(self, query: str) -> list[str]:

        logging.info("\nSplitting query")

        prompt = split_prompt

        with self.generative_lock:
            response = GENERATIVE_MODEL.invoke(prompt)
        logging.info(f"Subqueries in split_query: {response}")

        if isinstance(response, str):
            start = response.find("[")
            end = response.rfind("]") + 1
            subqueries = json.loads(response[start:end])
        else:
            subqueries = [query]

        return subqueries

    def finalize(self, query: str, subqueries: list[str], answers: list[str]) -> dict:

        logging.info("\nFinalizing")

        prompt = finalize_prompt

        with self.generative_lock:
            response = GENERATIVE_MODEL.invoke(prompt)
        logging.info(f"Response in finalize: {response}")

        final_answer = (
            response
            if response
            else "Sorry, we are unable to \
            provide an overall feedback due to lack of relevant data."
        )

        return final_answer

    def run_subtask(self, subquery: str) -> dict:
        # Run specific task
        result = asyncio.run(self.invoke_langgraph(subquery))
        return result

    def manage_subtasks(self, query: str) -> dict:

        # Manage multiple calls or subqueries

        subqueries = self.split_query(query)

        with ThreadPoolExecutor(max_workers=len(subqueries)) as worker:
            results = list(worker.map(self.run_subtask, subqueries))

        answers = []
        for id, result in enumerate(results):
            if isinstance(result, Exception):
                answers.append(
                    f"Error in subquery {subqueries[id]}: \
                    {str(result)}"
                )
            else:
                answers.append(
                    result.get("answer", "No answer generated for this subquery.")
                )

        concatenated_answer = self.finalize(query, subqueries, answers)

        return {"result": concatenated_answer, "states": results}

    async def answer_question(self, query: str) -> Any:
        start = time.time()
        result = self.manage_subtasks(query)
        end = time.time()
        logging.info(f"answer_question: {end-start}")

        return result


async def main():
    agent = GNQNA(corpus_path=CORPUS_PATH, pcorpus_path=PCORPUS_PATH, db_path=DB_PATH)

    output = await agent.answer_question(question)
    logging.info("\nSystem feedback:", output["result"])

    GENERATIVE_MODEL.client.close()
    SUMMARY_MODEL.client.close()


if __name__ == "__main__":
    asyncio.run(main())
