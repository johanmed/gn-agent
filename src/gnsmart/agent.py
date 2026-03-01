"""
Smarter multi-agent system for GeneNetwork based on tool calling only
This is the main module of the package
To run: `python agent.py`
Author: Johannes Medagbe
Copyright (c) 2026
"""

import asyncio
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel
from typing_extensions import Annotated

from gnsmart.config import *
from gnsmart.prompts import *

warnings.filterwarnings("ignore")


class AgentState(BaseModel):
    """
    Represents agent state
    Avails 02 attributes to allow communication between agents
    """

    messages: Annotated[list[BaseMessage], add_messages]
    next_decision: Literal["researcher", "planner", "reflector", "expert", "end"]


@dataclass
class GNAgent:
    """
    Represent GeneNetwork Multi-Agent System
    Encapsulate all functionalities of the system
    Take:
         All actors prompts
         Default parameters:
             max_global_visits: maximum number of redirections allowed to prevent infinite looping
    Execute operations:
         Initialization of multi-agent graph
         Run of query through system
    """

    supervisor_prompt1: Any
    supervisor_prompt2: Any
    planner_prompt: Any
    researcher_prompt: Any
    expert_prompt: Any
    reflector_prompt: Any
    max_global_visits: int = 10

    def supervisor(self, state: AgentState) -> dict:
        """Manages interactions between other agents in system

        Args:
            state: agent state with relevant data

        Returns:
            agent state updated with next agent to call
        """

        logging.info("Supervising")
        messages = [
            ("system", self.supervisor_prompt1),
            *state.messages,
            ("system", self.supervisor_prompt2),
        ]

        if len(messages) > self.max_global_visits:
            return {"next_decision": "end"}

        result = supervise(background=messages)
        logging.info(f"Result in supervisor: {result}")
        next_decision = result.get("next_decision")

        return {
            "next_decision": next_decision,
        }

    def planner(self, state: AgentState) -> dict:
        """Plans steps to tackle a problem

        Args:
            state: agent state specifying problem

        Returns:
            agent state updated with plan
        """

        logging.info("Planning")
        input_text = [self.planner_prompt] + state.messages
        logging.info(f"Input in planner: {input_text}")
        result = plan(background=input_text)
        logging.info(f"Result in planner: {result}")
        answer = result.get("answer")

        return {
            "messages": [answer],
        }

    def researcher(self, state: AgentState) -> dict:
        """Address a query using model thinking and GeneNetwork SPARQL data accessed through ReAct tool calling

        Args:
            state: agent state containing query

        Returns:
            agent state updated with answer
        """

        logging.info("Researching")
        if len(state.messages) < 3:  # handle first call to researcher
            input_text = state.messages[0] + state.messages[1]  # use query and plan
        else:
            input_text = state.messages[-1]  # use reflection insights

        input_text = [self.researcher_prompt, input_text]
        logging.info(f"Input in researcher: {input_text}")

        research = ReactResearcher()
        result = research(query=input_text)

        logging.info(f"Result from researcher: {result}")
        answer = result.get("solution")

        return {
            "messages": [answer],
        }

    def expert(self, state: AgentState) -> dict:
        """Address a query using model thinking and NCBI search tool through ReAct

        Args:
            state: agent state containing query

        Returns:
            agent state updated with answer
        """

        logging.info("Expert extracting knowledge")
        if len(state.messages) < 4:  # handle first call to expert
            input_text = state.messages[0] + state.messages[1]
        else:
            input_text = state.messages[-2]

        input_text = [self.expert_prompt, input_text]
        logging.info(f"Input in expert: {input_text}")

        expertise = ReactExpert()
        result = expertise(query=input_text)

        logging.info(f"Result from expert: {result}")
        answer = result.get("solution")

        return {
            "messages": [answer],
        }

    def reflector(self, state: AgentState) -> dict:
        """Reflects about progress

        Args:
            state: agent state with current progress

        Returns:
            agent state updated with suggestions
        """

        logging.info("Reflecting")
        trans_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
        translated_messages = [self.reflector_prompt, state.messages[0]] + [
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

    def initialize_globgraph(self) -> Any:
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("researcher", self.researcher)
        graph_builder.add_node("planner", self.planner)
        graph_builder.add_node("reflector", self.reflector)
        graph_builder.add_node("supervisor", self.supervisor)
        graph_builder.add_node("expert", self.expert)
        graph_builder.add_edge(START, "planner")
        graph_builder.add_edge("researcher", "supervisor")
        graph_builder.add_edge("expert", "supervisor")
        graph_builder.add_edge("planner", "researcher")
        graph_builder.add_edge("reflector", "researcher")
        graph_builder.add_conditional_edges(
            "supervisor",
            lambda state: state.next_decision,
            {
                "reflector": "reflector",
                "researcher": "researcher",
                "expert": "expert",
                "end": END,
            },
        )
        graph = graph_builder.compile()

        return graph

    async def invoke_globgraph(self, query: str) -> Any:
        graph = self.initialize_globgraph()
        initial_state = {
            "messages": [("human", query)],
            "next_decision": "planner",  # always plan first
        }
        result = await graph.ainvoke(initial_state)
        return result

    async def handler(self, query: str) -> str:
        """
        Main question handler of the system
        """
        global_result = await self.invoke_globgraph(query)
        end_prompt = global_result.get("messages")
        end_result = end(messages=end_prompt)
        end_result = end_result.get("feedback")

        first_result = global_result.get("messages")[
            2
        ].content  # get first researcher feedback
        second_result = global_result.get("messages")[
            3
        ].content  # get first expert feedback

        output = f"\nInternal feedback: {first_result}\n\nExternal feedback: {second_result}\n\nProcessed feedback: {end_result}"

        return output


async def main(query: str) -> str:
    agent = GNAgent(
        supervisor_prompt1=supervisor_prompt1,
        supervisor_prompt2=supervisor_prompt2,
        planner_prompt=planner_prompt,
        researcher_prompt=researcher_prompt,
        expert_prompt=expert_prompt,
        reflector_prompt=reflector_prompt,
    )
    output = await agent.handler(query)
    logging.info(f"\n\nSystem feedback: {output}")

    return output


if __name__ == "__main__":
    logging.basicConfig(
        filename="log_smartagent.txt",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )
    asyncio.run(main(QUERY))
