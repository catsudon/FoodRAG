from typing import List
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool, Tool, tool
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from chromadb import ClientAPI
from ..media import PDFLoader, ImageLoader
from langchain import hub
from langgraph.prebuilt import create_react_agent


class BaseVectorDatabaseToolkit:
    def __init__(self, vector_database_client: ClientAPI, llm=ChatOpenAI(temperature=0.1), embedder=OpenAIEmbeddings(), prompt=None, collection_name: str = "food"):
        self.__vector_db_client = vector_database_client
        self.__llm = llm
        self.__embedder = embedder
        self.__prompt = prompt
        self.__pdf_loader = PDFLoader()
        self.__image_loader = ImageLoader()
        self.__vector_db = Chroma(client=self.__vector_db_client,
                                  embedding_function=self.__embedder,
                                  collection_name=collection_name)
        self.__vector_stuff_chain = create_stuff_documents_chain(
            llm=llm, prompt=hub.pull(
                "langchain-ai/retrieval-qa-chat")
        )
        self.__retrieval_chain = create_retrieval_chain(
            self.__vector_db.as_retriever(), self.__vector_stuff_chain)
        self.__agent = create_react_agent(
            model=self.__llm, tools=self.get_tools(), messages_modifier=self.__prompt
        )

    def load_pdf(self, path: str):
        """
        Useful when you want to read and summarize a PDF document.
        Arg:
            path: (str) Path to the PDF file.
        Return:
            list: List of Document objects.
        """
        docs = self.__pdf_loader.load(path, True)
        return docs

    def load_image(self, path: str):
        """
        Useful when you want to read and scan the context of image.
        Arg:
            path: (str) Path to the image file.
        Return:
            (Document, Image): Document object and PIL Image object.
        """
        doc, image = self.__image_loader.load(path)
        return doc, image

    def store_document(self, document: Document | list[Document]):
        """
        Useful when you want to store a document in the vector database.
        Not use this method to retrieve the document.

        Arg:
            document: (Document) Document object to store.
        """
        self.__vector_db.add_documents(
            document if isinstance(document, list) else [document])

    def store_image(self, image: str | list[str]):
        """
        Useful when you want to store an image in the vector database.
        Arg:
            image: (Image) PIL Image object.
        """
        self.__vector_db.add_images(
            image if isinstance(image, list) else [image])

    def retrieve_document(self, query: str, limit: int = 5):
        """
        Useful when you want to retrieve documents from the vector database.

        Arg:
            query: (str) Query to retrieve documents.
            limit: (int) Number of documents to retrieve.
        Return:
            list: List of Document objects.
        """
        return self.__retrieval_chain.invoke({"input": query, "limit": limit})

    def get_tools(self) -> List[BaseTool]:
        return self.get_retriver() + self.get_actionor()

    def get_sensitive_tools(self) -> List[BaseTool]:
        return self.get_actionor()

    def get_non_sensitive_tools(self) -> List[BaseTool]:
        return self.get_retriver()

    def get_retriver(self):
        @tool(parse_docstring=True)
        def retrieve_document(query: str, limit: int = 5):
            """ Useful when you want to retrieve documents from the vector database.

            Args:
                query: (str) Query to retrieve documents.
                limit: (int) Number of documents to retrieve.

            Returns:
                list: List of Document objects.
            """
            return self.retrieve_document(query, limit)
        return [retrieve_document]

    def get_actionor(self):
        @tool(parse_docstring=True)
        def load_pdf(path: str):
            """ Useful when you want to read and summarize a PDF document.

            Args:
                path: (str) Path to the PDF file.

            Returns:
                list: List of Document objects.
            """
            return self.load_pdf(path)

        @tool(parse_docstring=True)
        def load_image(path: str):
            """ Useful when you want to read and scan the context of image.

            Args:
                path: (str) Path to the image file.

            Returns:
                (Document, Image): Document object and PIL Image object.
            """
            return self.load_image(path)

        @tool(parse_docstring=True)
        def store_document(document: Document | list[Document]):
            """
            Useful when you want to store a document in the vector database.
            Not use this method to retrieve the document.

            Args:
                document: (Document) Document object to store.
            """
            self.store_document(document)

        @tool(parse_docstring=True)
        def store_image(image: str | list[str]):
            """
            Useful when you want to store an image in the vector database.

            Args:
                image: (Image) PIL Image object.
            """
            self.store_image(image)
        return [load_pdf, load_image, store_document, store_image]

    def invoke(self, state: dict, config=None):
        return self.__agent.invoke(state, config)
