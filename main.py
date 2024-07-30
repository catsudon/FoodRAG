import chainlit as cl
from chainlit.types import ThreadDict
import uuid
import json
from chromadb import HttpClient
from chromadb.config import Settings
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from app.components.core import GraphState, FileUploaded
from app.components.database import BaseDatabaseToolkit, BaseVectorDatabaseToolkit
from app.components.agent import FoodDataAgent


chat_memory = SqliteSaver.from_conn_string("database/chat_memory.db")


@cl.cache
def get_food_agent():
    food_chroma_client = HttpClient(
        host="chroma", settings=Settings(anonymized_telemetry=False))
    food_db = BaseDatabaseToolkit(
        sql_database_url="sqlite:///database/usda.db")
    food_vector = BaseVectorDatabaseToolkit(food_chroma_client)
    return FoodDataAgent(food_db, food_vector)


@cl.on_chat_start
def on_chat_start():
    food_agent = get_food_agent()
    builder = StateGraph(GraphState)
    builder.add_node("FOOD_DATA_AGENT", food_agent)
    builder.add_node("tools", ToolNode(food_agent.get_tools()))
    builder.add_edge(START, "FOOD_DATA_AGENT")
    builder.add_edge("FOOD_DATA_AGENT", END)
    builder.add_conditional_edges("FOOD_DATA_AGENT", tools_condition)
    builder.add_edge("tools", "FOOD_DATA_AGENT")
    builder.add_edge("tools", END)
    thread_id = str(uuid.uuid4())
    assistant = builder.compile(checkpointer=chat_memory)
    cl.user_session.set("assistant", assistant)
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("config", {"configurable": {"thread_id": thread_id}})
    cl.user_session.set("messages", set())


@cl.on_message
async def on_message(user_inp: cl.Message):
    assistant: CompiledGraph = cl.user_session.get("assistant")
    config = cl.user_session.get("config")

    uploaded_files = []
    if user_inp.elements:
        uploaded_files.append(FileUploaded(name=user_inp.elements[0].name,
                                           path=user_inp.elements[0].path,
                                           size=user_inp.elements[0].size,
                                           type=user_inp.elements[0].type,
                                           url=user_inp.elements[0].url))

    events = assistant.stream({"messages": ("user", user_inp.content), "uploaded_files": uploaded_files},
                              config, stream_mode="values")
    chat_history = cl.user_session.get("messages")
    print(user_inp.elements)
    async with cl.Step("query assistant to respond to user input", type="run") as run_step:
        run_step.input = user_inp.content
        for event in events:
            for message in event.get("messages", []):
                if message.id not in chat_history:
                    chat_history.add(message.id)
                    if isinstance(message, AIMessage):
                        await cl.Message(content=message.content, author="assistant").send()
                        print(message)
                        if message.tool_calls:
                            async with cl.Step(", ".join([tool["name"] for tool in message.tool_calls]), type="tool") as tool_step:
                                tool_step.input = message.tool_calls
                    if isinstance(message, ToolMessage):
                        tool_call_content = json.loads(message.content)
                        tool_step.output = tool_call_content['answer']
        run_step.output = "Assistant response complete"
    snapshot = assistant.get_state(config)
    print(snapshot)
    while snapshot.next:
        print(snapshot)
        snapshot = assistant.get_state(config)
    print("Snapshot ended!!")
    cl.user_session.set("messages", chat_history)


@cl.on_chat_resume
def on_chat_resume(thread: ThreadDict):
    thread_id = thread.get("id")
    cl.user_session.set("config", {"configurable": {"thread_id": thread_id}})
    if cl.user_session.get("assistant"):
        assistant = cl.user_session.get("assistant")
    cl.user_session.set("assistant", assistant)
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("messages", set())
    print("Chat resumed!!")
