from lingua import LanguageDetector
from deep_translator import GoogleTranslator
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableGenerator, RunnableConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from typing_extensions import Union
from ..core.state import TranslationState, SupervisorAgentState
from ..prompt import CONTEXT_TRANSLATOR_PROMPT_TEMPLATE


class ContextTranlatorNVIDIA(RunnableGenerator):
    def __init__(
        self,
        llm: ChatNVIDIA = ChatNVIDIA(temperature=0.1),
        target_language: str = "en",
        prompt: PromptTemplate | None = None,
    ):
        self.__llm = llm
        self.__translator = GoogleTranslator(
            source="auto", target=target_language)
        if prompt is None:
            prompt = PromptTemplate(
                template=CONTEXT_TRANSLATOR_PROMPT_TEMPLATE,
                input_variables=["original", "language", "translated"],
            ).partial(language=target_language)
        self.__prompt = prompt
        self.__translator = self.__prompt | self.__llm | StrOutputParser()

    def invoke(self, state: TranslationState, config: RunnableConfig = None):
        state["translated"] = self.__translator.translate(state["original"])
        state["revised"] = self.__translator.invoke(state, config)
        return state["revised"]


class ContextTranlatorOpenAI(RunnableGenerator):
    def __init__(
        self,
        llm: ChatOpenAI = ChatOpenAI(model="gpt-4o-mini", temperature=0.1),
        target_language: str = "en",
        prompt: PromptTemplate | None = None,
    ):
        self.__llm = llm
        self.__translator = GoogleTranslator(
            source="auto", target=target_language)
        if prompt is None:
            prompt = PromptTemplate(
                template=CONTEXT_TRANSLATOR_PROMPT_TEMPLATE,
                input_variables=["original", "language", "translated"],
            ).partial(language=target_language)
        self.__prompt = prompt
        self.__translator = self.__prompt | self.__llm | StrOutputParser()

    def invoke(self, state: TranslationState, config: RunnableConfig = None):
        state["translated"] = self.__translator.translate(state["original"])
        state["revised"] = self.__translator.invoke(state, config)
        return state["revised"]


class LanguageDetectorTool(BaseTool):
    name = "language_detector"
    description = "Detects the language of the user's input text and translate in to English for specialized agents."

    def __init__(self, detector: LanguageDetector = LanguageDetector(), translator: GoogleTranslator = GoogleTranslator()):
        self.__detector = detector
        self.__translator = translator

    def _run(self, state: SupervisorAgentState, config: RunnableConfig = None):
        last_message = state['message'][-1]
        if isinstance(last_message, HumanMessage):
            language = self.__detector.detect_language_of(last_message.content)
            translated = self.__translator.translate(last_message.content)
            state["translation"].append(
                TranslationState(
                    message_id=last_message.id,
                    original=last_message.content,
                    language=language,
                    translated=translated,
                )
            )
            if language != "en":
                state["message"].append(
                    ToolMessage(
                        tool_name="language_detector",
                        tool_calls=[],
                        content=f"Detected language: {
                            language}. Don't forget to translate back before respond to the user.",
                    ),
                    HumanMessage(
                        content=translated,
                        kwargs={"original": last_message.content,
                                "language": language, "translated": translated,
                                "ref_message_id": last_message.id},
                    )
                )
        return state


class LanguageTranslateBackTool(BaseTool):


if __name__ == "__main__":

    print("Context Translator")

    original = "The United States was founded in 1777 by a group of British colonists who wanted to break away from British rule."
    translated = "Estados Unidos fue fundado en 1777 por un grupo de colonos británicos que querían separarse del dominio británico."
    state = {"original": original, "language": "en", "translated": translated}

    translator = ContextTranlatorNVIDIA()
    result = translator.invoke(state)
    print(result)
    translator = ContextTranlatorOpenAI()
    result = translator.invoke(state)
    print(result)
