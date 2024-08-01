from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain.docstore.document import Document


class FileUploaded(TypedDict):
    name: str
    size: int
    type: str
    url: str | None
    path: str | None


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    generation: str
    question: str
    user_info: str
    chat_language: str | None
    documents: list[Document]
    translation: str
    uploaded_files: list[FileUploaded]
    error: list[str]


class TranslationState(TypedDict):
    message_id: str
    original: str
    language: str | None
    translated: str | None


class DocumentRetrivalState(TypedDict):
    documents: list[Document]


class PrompterGeneratorState(TypedDict):
    context: str
    question: str


class SupervisorAgentState(TypedDict):
    message: Annotated[list[AnyMessage], add_messages]
    translation: Annotated[list[TranslationState], add_messages]


class FoodAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    result: str
    documents: list[Document]
    error: str
