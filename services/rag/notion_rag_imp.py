import os
from langchain.document_loaders import NotionDBLoader
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from helpers import splitter
from services.rag.rag_service import RAGService


class NotionRAGImp(RAGService):
    """Class to handle data operations for Notion vector storage.
    This class is responsible for loading, processing, and saving data from Notion.
    It also provides methods to retrieve and summarize documents.
    """
    _vector_store = None

    def __init__(self, data_source):
        if self._vector_store is None:
            self._vector_store = data_source
        self._load_data()
        self._process_data()

    def _load_data(self):
        # Logic to load data from the Notion database
        token = os.getenv("NOTION_INTEGRATION_TOKEN")
        id = os.getenv("NOTION_DATABASE_ID")
        if not token or not id:
            raise ValueError(
                "Environment variables NOTION_INTEGRATION_TOKEN and NOTION_DATABASE_ID must be set.")
        loader = NotionDBLoader(
            integration_token=token,
            database_id=id
        )
        self.docs = loader.load()

    def _process_data(self):
        # Logic to process the loaded data
        text_splitter = splitter.RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        splits = text_splitter.split_documents(self.docs)
        if not splits:
            raise Exception("No splits generated from documents.")
        # or another embedding provider as appropriate
        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        self._vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="vector_store"  # Ensure this directory exists or is created
        )

    def get_retriever(self) -> VectorStoreRetriever:
        """Returns a retriever for the vector store."""
        if self._vector_store is None:
            raise Exception("Vector store is not initialized.")
        return self._vector_store.as_retriever()
