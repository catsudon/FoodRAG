from typing_extensions import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from ..core.state import SupervisorAgentState
from ..core.configuration import SupervisorAgentConfig
from ..prompt import SUPERVISOR_AGENT_SYSTEM_PROPMT_TEMPLATE


class SupervisorAgentRAG:
    def __init__(self, supervisor_llm=ChatOpenAI(temperature=0.1), supervisor_prompt=None, supervisor_tools: Optional[list[BaseTool]] = None) -> None:
        self.__supervisor_llm = supervisor_llm
        self.__supervisor_prompt = ChatPromptTemplate.from_messages(
            [("system", supervisor_prompt if supervisor_prompt else SUPERVISOR_AGENT_SYSTEM_PROPMT_TEMPLATE), ("placeholder", "{messages}")])
        self.__supervisor_tools = supervisor_tools
        self.__supervisor = self.__supervisor_prompt | self.__supervisor_llm.bind_tools(
            self.__supervisor_tools)

    def __call__(self, state: SupervisorAgentState, config: SupervisorAgentConfig) -> dict:
        messages = state["messages"]
        while True:
            result = self.__supervisor.invoke(state, config=config)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + \
                    [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
