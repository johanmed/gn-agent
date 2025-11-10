"""
Multi-agent system for GeneNetwork
This is the main module of the package
To run: `python agent.py`
Author: Johannes Medagbe
Copyright (c) 2025
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Literal

from gnagent.config import *
from gnagent.prompts import *
from gnagent.query import query
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from rdflib import Graph
from tqdm import tqdm
from typing_extensions import Annotated, TypedDict


class AgentState(BaseModel):
    """
    Represents agent state
    Avails 02 attributes to allow communication between agents
    """

    messages: Annotated[list[BaseMessage], add_messages]
    next: Literal["researcher", "planner", "reflector", "end"]


class SubagentState(TypedDict):
    """
    Represents state of subcomponents of the agent researcher
    Avails 05 attributes to allow communication between its subcomponents
    """

    input: str
    chat_history: list[str]
    context: list[str]
    answer: str
    should_continue: str


@dataclass
class GNAgent:
    """
    Represents GeneNetwork Agent
    Encapsulates all functionalities of the system
    Takes:
         Paths of corpuses and database
         All actors prompts
         Default parameters:
             chat_id: identifier of conversation thread
             max_global_visits: maximum number of redirections allowed to prevent infinite looping
    Executes operations:
         Document processing (including naturalization) if not yet done
         Document embedding and database creation if not yet done
         Initialization of multi-agent graph
         Initialization of subagent graph for researcher agent
         Run of query through system
    """

    corpus_path: str
    pcorpus_path: str
    db_path: str
    naturalize_prompt: Any
    rephrase_prompt: Any
    analyze_prompt: Any
    check_prompt: Any
    summarize_prompt: Any
    synthesize_prompt: Any
    split_prompt: Any
    finalize_prompt: Any
    sup_prompt1: Any
    sup_prompt2: Any
    plan_prompt: Any
    refl_prompt: Any
    max_global_visits: int = 5
    chroma_db: Any = field(init=False)
    docs: list = field(init=False)
    ensemble_retriever: Any = field(init=False)
    memory: Any = field(init=False)
    subgraph: Any = field(init=False)

    def __post_init__(self):

        # Process or load documents
        if not Path(self.pcorpus_path).exists():  # first time execution
            self.docs = self.corpus_to_docs(self.corpus_path)
            with open(self.pcorpus_path, "w") as file:
                file.write(json.dumps(self.docs))
        else:  # subsequent executions
            with open(self.pcorpus_path) as file:
                data = file.read()
                self.docs = json.loads(data)

        # Create or get embedding database
        self.chroma_db = self.set_chroma_db(
            docs=self.docs,
            embed_model=HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"trust_remote_code": True, "device": "cpu"},
            ),  # could use gpu instead of cpu with more RAM
            db_path=self.db_path,
        )

        # Init the ensemble retriever
        metadatas = [{"source": f"Document {ind + 1}"} for ind in range(len(self.docs))]
        bm25_retriever = BM25Retriever.from_texts(
            texts=self.docs,
            metadatas=metadatas,
            k=10,  # might need finetuning
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.chroma_db.as_retriever(
                    search_kwargs={"k": 10}
                ),  # might need finetuning
                bm25_retriever,
            ],
            weights=[0.5, 0.5],  # might need finetuning
            c=30,
        )
        self.memory = MemorySaver()
        self.subgraph = self.initialize_subgraph()

    def corpus_to_docs(
        self,
        corpus_path: str,
        chunk_size: int = 1,  # small chunk size to match embedding chunks
        make_natural: bool = False,
    ) -> list:
        """Extracts documents from file and performs processing

        Args:
            corpus_path: path to directory containing corpus
            chunk_size: minimal number of documents by iteration

        Returns:
            processed document chunks
        """

        logging.info("In corpus_to_docs")

        if not Path(corpus_path).exists():
            sys.exit(1)

        # Read documents from a single file in corpus path
        with open(f"{corpus_path}aggr_rdf.txt") as f:
            aggregated = f.read()
            collection = json.loads(aggregated)  # dictionary with key being RDF subject

        docs = []
        chunks = []

        for key in tqdm(collection):
            concat = ""
            for value in collection[key]:
                text = f"{key} is/has {value}. "
                concat += text
            chunks.append(concat)

        if make_natural == False:
            return chunks

        prompts = []
        for i in range(0, len(chunks) + 1, chunk_size):
            chunk = chunks[i : i + chunk_size]
            text = "".join(chunk)
            prompt = [self.naturalize_prompt.copy(), HumanMessage(text)]
            prompts.append(prompt)

        def naturalize(data: str) -> str:
            """Naturalizes RDF data

            Args:
                data: RDF triples

            Returns:
                logic text capturing RDF meaning
            """

            response = naturalize_pred(input=data)
            return response.get("answer")

        with ThreadPoolExecutor(max_workers=100) as ex:  # Explain magic number
            for answer in tqdm(ex.map(naturalize, prompts), total=len(prompts)):
                docs.append(answer)
            # Save on disk for quick turnaround
            with open(f"{corpus_path}proc_aggr_rdf.txt", "w") as f:
                f.write(json.dumps(docs))

        return docs

    def set_chroma_db(
        self, docs: list, embed_model: Any, db_path: str, chunk_size: int = 1
    ) -> Any:  # small chunk_size for atomicity and memory management
        """Initializes or reads embedding database

        Args:
            docs: processed document chunks
            embed_model: model for embedding
            db_path: path to database
            chunk_size: number of chunks to process by iteration

        Returns:
            database object for embedding
        """

        logging.info("In set_chroma_db")

        if Path(db_path).exists():
            db = Chroma(persist_directory=db_path, embedding_function=embed_model)
            return db
        else:
            db = Chroma(
                embedding_function=embed_model,
                persist_directory=db_path,
            )
            for i in tqdm(range(0, len(docs), chunk_size)):
                chunk = docs[i : i + chunk_size]
                metadatas = [
                    {"source": f"Document {ind + 1}"}
                    for ind in range(i, i + len(chunk))
                ]
                db.add_texts(
                    texts=chunk,
                    metadatas=metadatas,
                )

            db.persist()
            return db

    def rephrase(self, state: SubagentState) -> dict:
        """Rephrases a query to use information in memory

        Args:
            state: node state

        Returns:
            node state updated with memory
        """

        logging.info("Rephrasing")

        existing_history = (
            "\n".join(state.get("chat_history", []))
            if state.get("chat_history", [])
            else "No prior conversation."
        )

        rephrase_prompt = [self.rephrase_prompt.copy(), HumanMessage(state["input"])]

        response = rephrase_pred(
            input=rephrase_prompt, existing_history=[HumanMessage(existing_history)]
        )

        logging.info(f"Response in rephrase: {response}")

        response = response.get("answer")
        should_continue = "retrieve"

        return {
            "input": response,
            "answer": state.get("answer", ""),
            "should_continue": should_continue,
            "chat_history": state.get("chat_history", []),
            "context": state.get("context", []),
        }

    def retrieve(self, state: SubagentState) -> dict:
        """Retrieves relevant documents to a query

        Args:
            state: node state

        Returns:
            node state updated with retrieved documents
        """

        logging.info("Retrieving")

        logging.info(f"Input in retriever: {state['input']}")

        retrieved_docs = self.ensemble_retriever.invoke(state["input"]) + state.get(
            "context", []
        )

        logging.info(f"Retrieved docs in retrieve: {retrieved_docs}")

        should_continue = "analyze"

        return {
            "input": state["input"],
            "context": retrieved_docs,
            "should_continue": should_continue,
            "chat_history": state.get("chat_history", []),
            "answer": state.get("answer", ""),
        }

    def analyze(self, state: SubagentState) -> dict:
        """Addresses a query based on retrieved documents

        Args:
            state: node state

        Returns:
            node state updated with answer
        """

        logging.info("Analysing")

        context = (
            "\n".join(doc.page_content for doc in state.get("context", []))
            if state.get("context", [])
            else ""
        )

        truncated_context = str(context)[
            :25_000
        ]  # prehandle context length of large documents given model limit of 32_000

        existing_history = (
            "\n".join(state.get("chat_history", []))
            if state.get("chat_history", [])
            else ""
        )

        analyze_prompt = [self.analyze_prompt.copy(), HumanMessage(state["input"])]

        response = analyze_pred(
            input=analyze_prompt,
            context=[HumanMessage(truncated_context)],
            existing_history=[HumanMessage(existing_history)],
        )

        logging.info(f"Response in analyze: {response}")

        response = response.get("answer")
        should_continue = "check_relevance"

        return {
            "input": state["input"],
            "answer": response,
            "should_continue": should_continue,
            "context": state.get("context", []),
            "chat_history": state.get("chat_history", []),
        }

    def check_relevance(self, state: SubagentState) -> dict:
        """Checks relevance of answer to query

        Args:
            state: node state

        Returns:
            node state updated with relevance status
        """

        logging.info("Checking relevance")

        answer = state["answer"]

        check_prompt = [self.check_prompt.copy(), HumanMessage(state["input"])]

        assessment = check_pred(input=check_prompt, answer=[HumanMessage(answer)])
        logging.info(f"Assessment in checking relevance: {assessment}")

        if assessment.get("decision") == "yes":
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

    def summarize(self, state: SubagentState) -> dict:
        """Summarizes data in node

        Args:
            state: node state

        Returns:
            summarized answer
        """

        logging.info("Summarizing")

        current_interaction = f"""
            User: {state["input"]}\nAssistant: {state["answer"]}"""

        summarize_prompt = [
            self.summarize_prompt.copy(),
            HumanMessage(current_interaction),
        ]

        summary = summarize_pred(full_context=summarize_prompt)
        summary = summary.get("summary")

        if not summary or not isinstance(summary, str) or summary.strip() == "":
            summary = f"- {state['input']} - No valid answer generated"

        existing_history = state.get("chat_history", [])

        updated_history = existing_history + [summary]  # update chat_history
        logging.info(f"Chat history in summarize: {updated_history}")

        # Generate final answer
        if not updated_history:
            final_answer = "Insufficient data for analysis."
        else:
            synthesize_prompt = [
                self.synthesize_prompt.copy(),
                HumanMessage(state["input"]),
            ]
            result = synthesize_pred(
                input=synthesize_prompt, updated_history=[HumanMessage(updated_history)]
            )
            logging.info(f"Result in summarize: {result}")

            result = result.get("conclusion")
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

    def initialize_subgraph(self) -> Any:
        graph_builder = StateGraph(SubagentState)
        graph_builder.add_node("rephrase", self.rephrase)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("check_relevance", self.check_relevance)
        graph_builder.add_node("analyze", self.analyze)
        graph_builder.add_node("summarize", self.summarize)
        graph_builder.add_edge(START, "rephrase")
        graph_builder.add_edge("rephrase", "retrieve")
        graph_builder.add_edge("retrieve", "analyze")
        graph_builder.add_edge("analyze", "check_relevance")
        graph_builder.add_edge("summarize", END)
        graph_builder.add_conditional_edges(
            "check_relevance",
            lambda state: state.get("should_continue", "summarize"),
            {"summarize": "summarize", "end": END},
        )
        subgraph = graph_builder.compile(checkpointer=self.memory)

        return subgraph

    async def invoke_subgraph(self, question: str, thread_id: str) -> Any:

        config = {"configurable": {"thread_id": thread_id}}  # conversation thread
        result = await self.subgraph.ainvoke({"input": question}, config)

        return result

    def split_query(self, query: str) -> list[str]:

        logging.info("Splitting query")

        split_prompt = [self.split_prompt.copy(), HumanMessage(query)]
        result = subquery(query=split_prompt)

        logging.info(f"Subqueries in split_query: {result}")
        result = result.get("answer")

        return result

    def finalize(self, query: str, subqueries: list[str], answers: list[str]) -> dict:
        """Combines results of subqueries

        Args:
            query: original query
            subqueries: smaller queries
            answers: answers to smaller queries

        Returns:
            consensus result
        """

        logging.info("Finalizing")

        finalize_prompt = [self.finalize_prompt.copy(), HumanMessage(query)]
        result = finalize_pred(
            query=finalize_prompt,
            subqueries=[HumanMessage(subqueries)],
            answers=[HumanMessage(answers)],
        )

        logging.info(f"Result in finalize: {result}")
        result = result.get("conclusion")
        final_answer = (
            result
            if result
            else "Sorry, we are unable to \
            provide an overall feedback due to lack of relevant data."
        )

        return final_answer

    def run_subtask(self, subquery: str, research_thread_id: str) -> dict:
        # Handle a subquery
        result = asyncio.run(self.invoke_subgraph(subquery, research_thread_id))
        return result

    def manage_subtasks(self, query: str) -> dict:
        """Handles a query by decomposing it into smaller queries and
        answering them

        Args:
            query: original query

        Returns:
            final answer
        """

        research_thread_id = f"researcher_{uuid.uuid4().hex[:8]}"
        subqueries = self.split_query(query)

        answers = []
        for id, subquery in enumerate(subqueries):
            answer = self.run_subtask(subquery, research_thread_id)
            if isinstance(answer, Exception):
                answers.append(
                    f"Error in subquery {subqueries[id]}: \
                    {str(answer)}"
                )
            else:
                answers.append(
                    answer.get("answer", "No answer generated for this subquery.")
                )

        concatenated_answer = self.finalize(query, subqueries, answers)

        return concatenated_answer

    def researcher(self, state: AgentState) -> Any:
        """Researches a query

        Args:
            state: agent state containing query

        Returns:
            agent state updated with result
        """

        logging.info("Researching")
        start = time.time()
        if len(state.messages) < 3:
            input = state.messages[0]
        else:
            input = state.messages[-1]
        input = input.content
        logging.info(f"Input in researcher: {input}")
        result = self.manage_subtasks(input)
        end = time.time()
        logging.info(f"Result in researcher: {result}")

        return {
            "messages": [result],
        }

    def planner(self, state: AgentState) -> Any:
        """Plans steps to tackle a problem

        Args:
            state: agent state specifying problem

        Returns:
            agent state updated with plan
        """

        logging.info("Planning")
        input = [self.plan_prompt] + state.messages
        logging.info(f"Input in planner: {input}")
        result = plan(background=input)
        logging.info(f"Result in planner: {result}")
        answer = result.get("answer")
        return {
            "messages": [answer],
        }

    def reflector(self, state: AgentState) -> Any:
        """Reflects about progress

        Args:
            state: agent state with current progress

        Returns:
            agent state updated with suggestions
        """

        logging.info("Reflecting")
        trans_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
        translated_messages = [self.refl_prompt, state.messages[0]] + [
            trans_map[msg.__class__](content=msg.content) for msg in state.messages[1:]
        ]
        logging.info(f"Input in reflector: {translated_messages}")
        result = tune(background=translated_messages)
        logging.info(f"Result in reflector: {result}")
        answer = result.get("answer")
        answer = (
            "Progress has been made. Use now all the resources to addess this new suggestion: "
            + answer
        )
        return {
            "messages": [HumanMessage(answer)],
        }

    def supervisor(self, state: AgentState) -> Any:
        """Manages interactions between other agents in system

        Args:
            state: agent state with relevant data

        Returns:
            agent state updated with next agent to call
        """

        logging.info("Supervising")
        messages = [
            ("system", self.sup_prompt1),
            *state.messages,
            ("system", self.sup_prompt2),
        ]

        if len(messages) > self.max_global_visits:
            return {"next": "end"}

        result = supervise(background=messages)
        logging.info(f"Result in supervisor: {result}")
        next = result.get("next")

        return {
            "next": next,
        }

    def initialize_globgraph(self) -> Any:
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("researcher", self.researcher)
        graph_builder.add_node("planner", self.planner)
        graph_builder.add_node("reflector", self.reflector)
        graph_builder.add_node("supervisor", self.supervisor)
        graph_builder.add_edge(START, "planner")
        graph_builder.add_edge("researcher", "supervisor")
        graph_builder.add_edge("planner", "researcher")
        graph_builder.add_edge("reflector", "researcher")
        graph_builder.add_conditional_edges(
            "supervisor",
            lambda state: state.next,
            {
                "reflector": "reflector",
                "researcher": "researcher",
                "end": END,
            },
        )
        graph = graph_builder.compile()

        return graph

    async def invoke_globgraph(self, query: str) -> Any:
        graph = self.initialize_globgraph()
        initial_state = {
            "messages": [("human", query)],
            "next": "planner",  # always plan first
        }

        result = await graph.ainvoke(initial_state)

        return result

    async def handler(self, query: str) -> Any:
        # Main question handler of the system
        global_result = await self.invoke_globgraph(query)
        first_result = global_result.get("messages")[
            2
        ].content  # get first researcher feedback
        end_prompt = global_result.get("messages")
        end_result = end(question=end_prompt)
        end_result = (
            f"\nInitial: {first_result}\n\n Improved: {end_result.get('answer')}"
        )
        # Extract reasoning from all messages
        reasoning = " ".join(msg.content for msg in end_prompt)
        return end_result, reasoning


async def main(query: str):
    agent = GNAgent(
        corpus_path=CORPUS_PATH,
        pcorpus_path=PCORPUS_PATH,
        db_path=DB_PATH,
        naturalize_prompt=naturalize_prompt,
        rephrase_prompt=rephrase_prompt,
        analyze_prompt=analyze_prompt,
        check_prompt=check_prompt,
        summarize_prompt=summarize_prompt,
        synthesize_prompt=synthesize_prompt,
        split_prompt=split_prompt,
        finalize_prompt=finalize_prompt,
        sup_prompt1=sup_prompt1,
        sup_prompt2=sup_prompt2,
        plan_prompt=plan_prompt,
        refl_prompt=refl_prompt,
    )

    output = await agent.handler(query)
    logging.info(f"\n\nSystem feedback: {output}")


if __name__ == "__main__":
    asyncio.run(main(query))
