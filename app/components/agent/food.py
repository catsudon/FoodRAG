from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import tools_condition
from typing_extensions import Optional, Literal

from ..database import BaseDatabaseToolkit, BaseVectorDatabaseToolkit
from ..core.state import FoodAgentState
from ..prompt import FOOD_DATA_AGENT_SYSTEM_PROPMT_TEMPLATE, FOOD_DATA_MANAGER_AGENT_SYSTEM_PROPMT_TEMPLATE
from ..agent.assistant import Assistant
from ..utils.tools import create_tool_node_with_fallback


class FoodDataAgent(Runnable):
    def __init__(self, sql: BaseDatabaseToolkit, vector: BaseVectorDatabaseToolkit, llm=ChatOpenAI(temperature=0.1), checkpointer=None):
        self.__sql = sql
        self.__vector = vector
        self.__llm = llm
        self.__checkpointer = checkpointer
        self.__non_sensitive_tools = self.__sql.get_tools(
        ) + self.__vector.get_retriver()
        self.__workflow = StateGraph(FoodAgentState)
        self.__agent_prompt = ChatPromptTemplate.from_messages(
            [("system", FOOD_DATA_AGENT_SYSTEM_PROPMT_TEMPLATE), ("placeholder", "{messages}")])
        self.__agent = self.__agent_prompt | self.__llm.bind_tools(
            self.__non_sensitive_tools)
        self.__assistant = Assistant(self.__agent)
        self.__graph = self.__setup_workflow()

    def __generate_router(self):
        safe_toolnames = [t.name for t in self.__non_sensitive_tools]

        def route_food_agent(
            state: FoodAgentState,
        ) -> Literal[
            "query_tools",
            "__end__",
        ]:
            route = tools_condition(state)
            if route == END:
                return END
            return "query_tools"

        return route_food_agent

    def __setup_workflow(self):
        self.__workflow.add_node("assistant", self.__assistant)
        self.__workflow.add_node(
            "query_tools", create_tool_node_with_fallback(self.__non_sensitive_tools))
        self.__workflow.add_edge(START, "assistant")
        self.__workflow.add_edge("query_tools", "assistant")
        self.__workflow.add_conditional_edges(
            "assistant", self.__generate_router())
        return self.__workflow.compile(checkpointer=self.__checkpointer)

    def invoke(self, state: FoodAgentState, config: Optional[dict] = None, **kwargs) -> dict:
        return self.__graph.invoke(state, config=config, **kwargs)

    def get_sensitive_tools(self):
        return []

    def get_non_sensitive_tools(self):
        return self.__non_sensitive_tools

    def get_agent(self):
        return self.__agent

    def get_graph(self):
        return self.__graph


class FoodDataManagerAgent(Runnable):
    def __init__(self, sql: BaseDatabaseToolkit, vector: BaseVectorDatabaseToolkit, llm=ChatOpenAI(temperature=0.1), checkpointer=None):
        self.__sql = sql
        self.__vector = vector
        self.__llm = llm
        self.__checkpointer = checkpointer
        self.__sensitive_tools = self.__vector.get_actionor()
        self.__non_sensitive_tools = self.__sql.get_tools(
        ) + self.__vector.get_retriver()
        self.__workflow = StateGraph(FoodAgentState)
        self.__agent_prompt = ChatPromptTemplate.from_messages(
            [("system", FOOD_DATA_MANAGER_AGENT_SYSTEM_PROPMT_TEMPLATE), ("placeholder", "{messages}")])
        self.__agent = self.__agent_prompt | self.__llm.bind_tools(
            self.__non_sensitive_tools + self.__sensitive_tools)
        self.__assistant = Assistant(self.__agent)
        self.__graph = self.__setup_workflow()

    def __generate_router(self):
        safe_toolnames = [t.name for t in self.__non_sensitive_tools]

        def route_food_agent(
            state: FoodAgentState,
        ) -> Literal[
            "query_tools",
            "sensitive_tools",
            "__end__",
        ]:
            route = tools_condition(state)
            if route == END:
                return END
            tool_calls = state["messages"][-1].tool_calls
            if all(tc["name"] in safe_toolnames for tc in tool_calls):
                return "query_tools"
            return "sensitive_tools"

        return route_food_agent

    def __setup_workflow(self):
        self.__workflow.add_node("assistant", self.__assistant)
        self.__workflow.add_node(
            "query_tools", create_tool_node_with_fallback(self.__non_sensitive_tools))
        self.__workflow.add_node(
            "sensitive_tools", create_tool_node_with_fallback(self.__sensitive_tools))
        self.__workflow.add_edge(START, "assistant")
        self.__workflow.add_edge("query_tools", "assistant")
        self.__workflow.add_edge("sensitive_tools", "assistant")
        self.__workflow.add_conditional_edges(
            "assistant", self.__generate_router())
        return self.__workflow.compile(interrupt_before=["sensitive_tools"], checkpointer=self.__checkpointer)

    def invoke(self, state: FoodAgentState, config: Optional[dict] = None, **kwargs) -> dict:
        return self.__graph.invoke(state, config=config, **kwargs)

    def get_sensitive_tools(self):
        return self.__sensitive_tools

    def get_non_sensitive_tools(self):
        return self.__non_sensitive_tools

    def get_agent(self):
        return self.__agent

    def get_graph(self):
        return self.__graph
