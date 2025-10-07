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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from rdflib import Graph
from tqdm import tqdm
from typing_extensions import Annotated, TypedDict

from config import *
from prompts import *
from question import question


class State(TypedDict):
    input: str
    chat_history: list[str]
    context: list[str]
    answer: str
    should_continue: str


class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    next: Literal["researcher", "planner", "reflector", "end"]


@dataclass
class GNAgent:
    corpus_path: str
    pcorpus_path: str
    db_path: str
    max_global_visits: int = 15  # max visits allowed in the global graph
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
        metadatas = [{"source": f"Document {ind}"} for ind in range(len(self.docs))]
        bm25_retriever = BM25Retriever.from_texts(
            texts=self.docs,
            metadatas=metadatas,
            k=10,
        )
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

            with self.generative_lock:
                naturalize_prompt = naturalize_prompt.format(text=text)
                response = deep_generate(question=naturalize_prompt)
                response = response.get("answer")
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
            db = Chroma(
                embedding_function=embed_model,
                persist_directory=db_path,
            )
            for i in tqdm(range(0, len(docs) + 1, chunk_size)):
                chunk = docs[i : i + chunk_size]
                metadatas = [
                    {"source": f"Document {ind}"} for ind in range(i, i + len(chunk))
                ]
                db.add_texts(
                    texts=chunk,
                    metadatas=metadatas,
                )

            db.persist()
            return db

    def retrieve(self, state: State) -> dict:

        # Retrieve documents
        logging.info("Retrieving")

        with self.retriever_lock:
            retrieved_docs = self.ensemble_retriever.invoke(state["input"])

        logging.info(f"Retrieved docs in retrieve: {retrieved_docs}")

        should_continue = "analyze"

        return {
            "input": state["input"],
            "context": retrieved_docs,
            "should_continue": should_continue,
            "chat_history": state.get("chat_history", []),
            "answer": state.get("answer", ""),
        }

    def analyze(self, state: State) -> dict:

        # Analyze documents
        logging.info("Analysing")

        context = (
            "\n".join(doc.page_content for doc in state.get("context", []))
            if state.get("context", [])
            else ""
        )

        existing_history = (
            "\n".join(state.get("chat_history", []))
            if state.get("chat_history", [])
            else ""
        )

        with self.generative_lock:
            analyze_prompt = analyze_prompt.format(
                context=context, existing_history=existing_history, input=state["input"]
            )
            response = deep_generate(question=analyze_prompt)

        logging.info(f"Response in analyze: {response}")

        response = " ".join(response.get("answer").split(" ")[:200])  # constraint
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
        logging.info("Checking relevance")

        answer = state["answer"]

        with self.summary_lock:
            check_prompt = check_prompt.format(answer=answer, input=state["input"])
            assessment = shallow_generate(question=check_prompt)
        logging.info(f"Assessment in checking relevance: {assessment}")

        if "yes" in assessment.get("answer").lower():
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
        logging.info("Summarizing")

        existing_history = state.get("chat_history", [])

        current_interaction = f"""
            User: {state["input"]}\nAssistant: {state["answer"]}"""

        full_context = (
            "\n".join(existing_history) + "\n" + current_interaction
            if existing_history
            else current_interaction
        )

        with self.summary_lock:
            summarize_prompt = summarize_prompt.format(full_context=full_context)
            summary = shallow_generate(question=summarize_prompt)
            summary = summary.get("answer")

        if not summary or not isinstance(summary, str) or summary.strip() == "":
            summary = f"- {state['input']} - No valid answer generated"

        updated_history = existing_history + [summary]  # update chat_history
        logging.info(f"Chat history in summarize: {updated_history}")

        # Generate final answer
        if not updated_history:
            final_answer = "Insufficient data for analysis."
        else:
            with self.generative_lock:
                synthesize_prompt = synthesize_prompt.format(
                    input=state["input"], updated_history=updated_history
                )
                result = deep_generate(question=synthesize_prompt)
            logging.info(f"Result in summarize: {result}")

            result = result.get("answer")
            final_answer = (
                result
                if result
                else "Sorry, we are unable to \
            provide a valuable feedback due to lack of relevant data."
            )

        return {
            "input": state["input"],
            "answer": final_answer,
            "context": state.get("context", []),
            "chat_history": updated_history,
        }

    def initialize_subgraph(self, state: State) -> Any:
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
        subgraph = graph_builder.compile()

        return subgraph

    async def invoke_subgraph(self, question: str) -> Any:
        subgraph = self.initialize_subgraph()
        initial_state = {
            "input": question,
            "chat_history": [],
            "context": [],
            "answer": "",
            "should_continue": "retrieve",
        }

        result = await subgraph.ainvoke(initial_state)

        return result

    def split_query(self, query: str) -> list[str]:

        logging.info("Splitting query")

        with self.generative_lock:
            split_prompt = split_prompt.format(query=query)
            result = deep_generate(question=split_prompt)

        logging.info(f"Subqueries in split_query: {result}")
        result = result.get("answer")

        if isinstance(result, str):
            start = result.find("[")
            end = result.rfind("]") + 1
            subqueries = json.loads(result[start:end])
        else:
            subqueries = [query]

        return subqueries

    def finalize(self, query: str, subqueries: list[str], answers: list[str]) -> dict:

        logging.info("Finalizing")

        with self.generative_lock:
            finalize_prompt = finalize_prompt.format(
                query=query, subqueries=subqueries, answers=answers
            )
            result = deep_generate(question=finalize_prompt)

        logging.info(f"Result in finalize: {result}")
        result = result.get("answer")
        final_answer = (
            result
            if result
            else "Sorry, we are unable to \
            provide an overall feedback due to lack of relevant data."
        )

        return final_answer

    def run_subtask(self, subquery: str) -> dict:
        # Run specific task
        result = asyncio.run(self.invoke_subgraph(subquery))
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

        return concatenated_answer

    async def researcher(self, query: str) -> Any:
        logging.info("Researching")
        start = time.time()
        result = self.manage_subtasks(query)
        end = time.time()
        logging.info(f"Result in researcher: {result}")

        return {"messages": [result["result"]]}

    def planner(self, state: AgentState) -> Any:
        logging.info("Planning")
        result = process(background=[plan_system_prompt] + state["messages"])
        logging.info(f"Result in planner: {result}")
        answer = result.get("answer")
        return {"messages": [answer]}

    def reflector(self, state: AgentState) -> Any:
        logging.info("Reflecting")
        trans_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
        translated_messages = [refl_system_prompt, state["messages"][0]] + [
            trans_map[msg.__class__](msg) for msg in state["messages"][1:]
        ]
        result = process(background=translated_messages)
        logging.info(f"Result in reflector: {result}")
        answer = result.get("answer")
        return {"messages": [HumanMessage(answer)]}

    def supervisor(self, state: AgentState) -> Any:
        logging.info("Supervising")
        messages = [
            ("system", sup_system_prompt1),
            *state["messages"],
            ("system", sup_system_prompt2),
        ]
        if len(messages) >= self.max_global_visits:
            return END
        result = supervise(background=messages)
        logging.info(f"Result in supervisor: {result}")
        result = result.get("next")
        return {"next": result}

    def initialize_globgraph(self, state: AgentState) -> Any:
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("researcher", self.researcher)
        graph_builder.add_node("planner", self.planner)
        graph_builder.add_node("reflector", self.reflector)
        graph_builder.add_node("supervisor", self.supervisor)
        graph_builder.add_edge(START, "supervisor")
        graph_builder.add_edge("researcher", "supervisor")
        graph_builder.add_edge("planner", "supervisor")
        graph_builder.add_edge("reflector", "supervisor")
        graph_builder.add_edge("end", END)
        graph_builder.add_conditional_edges("supervisor", lambda state: state["next"])
        graph = graph_builder.compile()

        return graph

    def invoke_globgraph(self, query: str) -> Any:
        graph = initialize_globgraph()
        initial_state = {
            "messages": [("human", query)],
            "next": Literal["supervisor"],
        }
        result = graph.ainvoke(initial_state)

        return result

    def handler(self, query: str) -> Any:
        global_result = asyncio.run(self.invoke_globgraph(query))
        return global_result


async def main():
    agent = GNAgent(corpus_path=CORPUS_PATH, pcorpus_path=PCORPUS_PATH, db_path=DB_PATH)

    output = await agent.handler(question)
    logging.info("\nSystem feedback:", output)


if __name__ == "__main__":
    asyncio.run(main())
