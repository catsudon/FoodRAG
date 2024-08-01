from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

import dotenv
import os
from state import GraphState


def load_env():
    dotenv.load_dotenv()
    return os.environ


load_env()


@tool()
def get_user_info(state: GraphState):
    """
    Useful when we want to know the user's name or other information
    Arg:
      state (State): The current state of the conversation
    Return:
      str: The user's name
    """
    if state.get("user_info"):
        return "Mathran"
    else:
        return "User"


@tool()
def recommend_food(state: GraphState):
    """
    Useful when recommending food to the user
    Arg:
      state (State): The current state of the conversation
    Return:
      str: Food menu
    """
    if state.get("user_info"):
        return "Omlette"
    else:
        return "Salad"


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: GraphState, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            user_info = configuration.get("user_info", None)
            state = {**state, "user_info": user_info}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            print(result)
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


chat_tools = [get_user_info, recommend_food]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for FoodRAG Inc. "
            " Use the provided tools to search for food information, nutrients, patient information, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>",
        ),
        ("placeholder", "{messages}"),
    ]
)

chat_runnable = chat_prompt | llm.bind_tools(chat_tools)


builder = StateGraph(GraphState)
builder.add_node("assistant", Assistant(chat_runnable))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_node("tools", create_tool_node_with_fallback(chat_tools))
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
builder.add_edge("assistant", END)

memory = SqliteSaver.from_conn_string(":memory:")
chat_graph = builder.compile(checkpointer=memory)

tutorial_questions = [
    "Hi there, what is my name?",
    "What is the weather like today?",
    "Can you get my medical information",
    "What is the best food to eat for breakfast?",
]

config = {
    "configurable": {
        "thread_id": "1",
    }
}


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


_printed = set()
for question in tutorial_questions:
    events = chat_graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event, _printed)
