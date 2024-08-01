from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict


class UserInfoConfig(TypedDict):
    username: str


class SupervisorAgentConfig(RunnableConfig):
    user_info: UserInfoConfig
