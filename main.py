import chainlit as cl
from chainlit.types import ThreadDict
from chainlit.element import ElementBased
import uuid
import json
from chromadb import HttpClient
from chromadb.config import Settings
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from app.components.core.state import SupervisorAgentState, FileUploaded
from app.components.prompt import SUPERVISOR_AGENT_SYSTEM_PROPMT_TEMPLATE
from app.components.database import BaseDatabaseToolkit, BaseVectorDatabaseToolkit
from app.components.agent import FoodDataAgent, Assistant
from app.components.prebuilt import SupervisorAgentRAG

chat_memory = SqliteSaver.from_conn_string("database/chat_memory.db")


@cl.cache
def get_food_agent():
    food_chroma_client = HttpClient(
        host="chroma", settings=Settings(anonymized_telemetry=False))
    food_db = BaseDatabaseToolkit(
        sql_database_url="sqlite:///database/usda.db")
    food_vector = BaseVectorDatabaseToolkit(food_chroma_client)
    return FoodDataAgent(food_db, food_vector)


@cl.cache
def get_supervisor_agent():
    prompt = ChatPromptTemplate.from_messages(
        ("system", SUPERVISOR_AGENT_SYSTEM_PROPMT_TEMPLATE), ("placeholder", "{messages}"))
    return SupervisorAgentRAG(
        supervisor_prompt=prompt, supervisor_tools=[])


@cl.on_chat_start
def on_chat_start():
    food_agent = get_food_agent()

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



@cl.on_chat_start
async def start():
    await cl.Message(
        content="Welcome to the Chainlit audio example. Press `P` to talk!"
    ).send()





from io import BytesIO
from pydub import AudioSegment
from utils.webm2wav import convert_webm_to_wav
from utils.ASR import ASR

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        print("Starting new audio stream...")
        buffer = BytesIO()
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)
        print("Initialized buffer and stored in session.")

    # Retrieve the buffer and write the incoming chunk
    buffer = cl.user_session.get("audio_buffer")
    if buffer is not None:
        buffer.write(chunk.data)
        print(f"Appended chunk to buffer. Buffer size: {buffer.tell()} bytes.")
    else:
        print("Error: Buffer not found in session.")

@cl.on_audio_end
async def on_audio_end(elements: list):
    # Retrieve the audio buffer from the session
    audio_buffer = cl.user_session.get("audio_buffer")
    if audio_buffer is None:
        print("Error: Audio buffer is None.")
        return

    audio_buffer.seek(0)  # Rewind the buffer to the beginning

    # Save the buffer as a webm file
    output_filename = "tmp/output_audio.webm"
    try:
        with open(output_filename, "wb") as f:
            f.write(audio_buffer.getvalue())
        print(f"Audio saved successfully as '{output_filename}'.")
        convert_webm_to_wav("tmp/output_audio.webm", "tmp/output_audio.wav")

        transcribed_text = ASR("tmp/output_audio.wav")
        print(transcribed_text)

    except Exception as e:
        print(f"Error saving audio: {e}")