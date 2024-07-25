from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

from ..database import BaseDatabaseToolkit, BaseVectorDatabaseToolkit


FOOD_DATA_AGENT_SYSTEM_PROPMT_TEMPLATE = """
You are a helpful agent designed to interact with a Food Database and Vectorstore.
Use provided tools to search, store, and retrieve food-related documents, images, and information.
When searching, be persistent. Expand your query bounds if the first search returns no results.
If a search comes up empty, expand your search before giving up.
"""


class FoodDataAgent:
    def __init__(self, sql: BaseDatabaseToolkit, vector: BaseVectorDatabaseToolkit, llm=ChatOpenAI(temperature=0.1)):
        self.__sql = sql
        self.__vector = vector
        self.__llm = llm
        self.__tools = [*self.__sql.get_tools(), *self.__vector.get_tools()]
        self.__agent = create_react_agent(
            model=self.__llm, tools=self.__tools,
            messages_modifier=SystemMessage(
                content=FOOD_DATA_AGENT_SYSTEM_PROPMT_TEMPLATE)
        )

    def get_tools(self):
        return self.__tools

    def invoke(self, state, config=None):
        return self.__agent.invoke(state, config)
