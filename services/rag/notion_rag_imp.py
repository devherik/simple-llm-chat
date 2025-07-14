import os
from langchain_core.documents.base import Document
from typing import List
from langchain.document_loaders import NotionDBLoader
from helpers import splitter
from services.rag.rag_service import RAGService


class NotionRAGImp(RAGService):
    """Class to handle data operations for Notion vector storage.
    This class is responsible for loading, processing, and saving data from Notion.
    It also provides methods to retrieve and summarize documents.
    """
    docs: List[Document] = []

    async def __init__(self):
        await self._load_data()
        await self._process_data()

    async def _load_data(self) -> None:
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

    async def _process_data(self) -> None:
        # Logic to process the loaded data
        splits = splitter.splitter_documents(self.docs)
        if not splits:
            raise Exception("No splits generated from documents.")
        self.docs = splits
