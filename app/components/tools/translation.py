from deep_translator import GoogleTranslator
from langchain_core.runnables import RunnableGenerator, RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..core.state import TranslationState
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
