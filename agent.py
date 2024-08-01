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
    food_agent = get_food_agent().get_graph()
    food_agent.checkpointer = chat_memory
    thread_id = str(uuid.uuid4())
    cl.user_session.set("assistant", food_agent)
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

    events = assistant.stream({"messages": ("user", user_inp.content)},
                              config, stream_mode="values")
    chat_history = cl.user_session.get("messages")
    async with cl.Step("query assistant to respond to user input", type="run") as run_step:
        run_step.input = user_inp.content
        for event in events:
            for message in event.get("messages", []):
                if message.id not in chat_history:
                    chat_history.add(message.id)
                    if isinstance(message, AIMessage):
                        await cl.Message(content=message.content, author="assistant").send()
                        print("AI", message)
                    elif isinstance(message, ToolMessage):
                        async with cl.Step(message.name, type="tool") as tool_step:
                            tool_step.input = message.content
        run_step.output = "Assistant response complete"
    snapshot = assistant.get_state(config)
    print("Snapshot started!!")
    while snapshot.next:
        user = await cl.AskActionMessage("Do you want to continue the conversation?", [
            cl.Action("continue", "Yes"), cl.Action("stop", "No")], "assistant").send()
        if user == "stop":
            assistant.invoke({
                "messages": [
                    ToolMessage(
                        tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                        content=f"API call denied by user. Continue assisting, accounting for the user's input.",
                    )
                ]
            },
                config,)
        else:
            assistant.invoke(None, config)
        snapshot = assistant.get_state(config)
    print("Snapshot ended!!")
    cl.user_session.set("messages", chat_history)


@ cl.on_chat_resume
def on_chat_resume(thread: ThreadDict):
    thread_id = thread.get("id")
    cl.user_session.set("config", {"configurable": {"thread_id": thread_id}})
    if cl.user_session.get("assistant"):
        assistant = cl.user_session.get("assistant")
    cl.user_session.set("assistant", assistant)
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("messages", set())
    print("Chat resumed!!")
