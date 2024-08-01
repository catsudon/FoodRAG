from langchain_core.runnables import RunnableGenerator, RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.components.core.state import PrompterGeneratorState
from ..prompt import PROMPTER_PROMPT_TEMPLATE


class PrompterGeneratorNVIDIA(RunnableGenerator):
    def __init__(
        self,
        llm: ChatNVIDIA = ChatNVIDIA(temperature=0.1),
        prompt: PromptTemplate | None = None,
    ):
        self.__llm = llm
        if prompt is None:
            prompt = PromptTemplate(
                template=PROMPTER_PROMPT_TEMPLATE,
                input_variables=["context", "question"],
            )
        self.__prompt = prompt
        self.__generator = self.__prompt | self.__llm | StrOutputParser()

    def invoke(self, state: PrompterGeneratorState, config: RunnableConfig = None):
        return self.__generator.invoke(state, config)


class PrompterGeneratorOpenAI(RunnableGenerator):
    def __init__(
        self,
        llm: ChatOpenAI = ChatOpenAI(model="gpt-4o-mini", temperature=0.1),
        prompt: PromptTemplate | None = None,
    ):
        self.__llm = llm
        if prompt is None:
            prompt = PromptTemplate(
                template=PROMPTER_PROMPT_TEMPLATE,
                input_variables=["context", "question"],
            )
        self.__prompt = prompt
        self.__generator = self.__prompt | self.__llm | StrOutputParser()

    def invoke(self, state: PrompterGeneratorState, config: RunnableConfig = None):
        return self.__generator.invoke(state, config)
